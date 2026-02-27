import os
import json
import logging
import asyncio
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Импорты Autogen
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== НАСТРОЙКИ ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.error("❌ DEEPSEEK_API_KEY не найден в переменных окружения!")

MODEL_NAME = "deepseek-chat"
# ===================================================

# Глобальные переменные
model_client = None
agent_team = None

def init_model_client():
    """Инициализация клиента DeepSeek"""
    global model_client
    try:
        logger.info("🔄 Инициализация модели DeepSeek...")
        model_client = OpenAIChatCompletionClient(
            model=MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True
            }
        )
        logger.info("✅ Модель DeepSeek инициализирована")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации модели: {e}", exc_info=True)
        return False

def init_agent_team():
    """Создание команды агентов"""
    global agent_team, model_client
    try:
        logger.info("🔄 Создание команды агентов...")
        
        tech_analyst = AssistantAgent(
            name="Tech_Analyst",
            model_client=model_client,
            system_message="""Ты — технический аналитик с 10-летним опытом. 
            Анализируй только графики и индикаторы: RSI, MACD, уровни поддержки/сопротивления, тренды.
            Игнорируй новости и слухи. Отвечай кратко и по существу."""
        )
        
        news_analyst = AssistantAgent(
            name="News_Analyst",
            model_client=model_client,
            system_message="""Ты — новостной аналитик. Оценивай рыночные настроения на основе новостей.
            Используй шкалу от -1 (медвежий) до +1 (бычий). Отмечай резкие изменения сентимента."""
        )
        
        decision_maker = AssistantAgent(
            name="Decision_Maker",
            model_client=model_client,
            system_message="""Ты — главный аналитик, принимающий решения. 
            Собери отчеты от технического и новостного аналитиков.
            Взвесь их аргументы и дай итоговую торговую рекомендацию.
            Укажи: направление (BUY/SELL/HOLD), уровни входа, take-profit, stop-loss."""
        )
        
        agent_team = RoundRobinGroupChat(
            participants=[tech_analyst, news_analyst, decision_maker],
            max_turns=5
        )
        logger.info("✅ Команда агентов создана")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка создания команды: {e}", exc_info=True)
        return False

async def run_analysis(task: str) -> str:
    """Запуск анализа задачи командой агентов"""
    try:
        logger.info(f"🔍 Начинаю анализ задачи: {task[:100]}...")
        
        result_parts = []
        async for message in agent_team.run_stream(task=task):
            if isinstance(message, TaskResult):
                continue
            if hasattr(message, 'content') and message.content:
                result_parts.append(f"**{message.source}**: {message.content}")
                logger.info(f"💬 {message.source}: {message.content[:100]}...")
        
        final_result = "\n\n".join(result_parts) if result_parts else "Анализ завершен, но агенты не дали ответа."
        logger.info(f"✅ Анализ завершен. Длина ответа: {len(final_result)} символов")
        return final_result
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа: {e}", exc_info=True)
        return f"❌ Ошибка при анализе: {str(e)}"

# ==================== МАРШРУТЫ ====================

@app.route('/', methods=['GET'])
def index():
    """Корневой маршрут - информация о сервисе"""
    return jsonify({
        "status": "CaptainAgent running",
        "version": "2.0",
        "endpoints": {
            "/": "GET - информация о сервисе",
            "/health": "GET - проверка здоровья",
            "/analyze": "POST - отправить задачу на анализ"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check для Render"""
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    """Основной маршрут для анализа задач"""
    logger.info("📥 POST /analyze called")
    
    try:
        # Проверяем инициализацию
        if not model_client or not agent_team:
            logger.error("❌ Система не инициализирована")
            return jsonify({"error": "System not initialized"}), 503
        
        # Получаем задачу из JSON
        data = request.get_json()
        if not data:
            logger.warning("⚠️ Нет JSON в запросе")
            return jsonify({"error": "No JSON data"}), 400
        
        task = data.get('task')
        if not task:
            logger.warning("⚠️ Нет поля 'task' в JSON")
            return jsonify({"error": "Missing 'task' field"}), 400
        
        logger.info(f"📝 Получена задача: {task[:200]}...")
        
        # Запускаем анализ в отдельном цикле событий
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_analysis(task))
        finally:
            loop.close()
        
        logger.info(f"✅ Отправляю результат клиенту")
        return jsonify({"result": result}), 200
        
    except Exception as e:
        logger.error(f"❌ Необработанная ошибка в /analyze: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# ==================== ЗАПУСК ====================

if __name__ == '__main__':
    if init_model_client() and init_agent_team():
        port = int(os.environ.get('PORT', 10000))
        logger.info(f"🚀 Запуск CaptainAgent на порту {port}")
        logger.info("🤖 Команда из 3 агентов готова к работе")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("❌ Не удалось инициализировать систему")
