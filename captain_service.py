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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== НАСТРОЙКИ ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.error("❌ DEEPSEEK_API_KEY не найден в переменных окружения!")
    # Можно здесь завершить работу или выставить флаг ошибки

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
            system_message="Ты — технический аналитик. Анализируй графики и индикаторы. Отвечай кратко, только по делу."
        )
        news_analyst = AssistantAgent(
            name="News_Analyst",
            model_client=model_client,
            system_message="Ты — новостной аналитик. Оценивай рыночные настроения на основе новостей. Будь краток."
        )
        decision_maker = AssistantAgent(
            name="Decision_Maker",
            model_client=model_client,
            system_message="Ты — главный аналитик. Собери отчеты от других агентов и дай итоговую рекомендацию."
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

def run_analysis_sync(task: str) -> str:
    """Синхронная обёртка для запуска асинхронного анализа."""
    try:
        logger.info(f"🔍 Начинаю синхронный анализ задачи: {task[:50]}...")
        # Создаём новый цикл событий для этого потока
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_parts = []
        try:
            # Запускаем асинхронную генерацию
            async def analyze():
                async for message in agent_team.run_stream(task=task):
                    if isinstance(message, TaskResult):
                        continue
                    if hasattr(message, 'content') and message.content:
                        part = f"{message.source}: {message.content}"
                        result_parts.append(part)
                        logger.info(f"💬 {message.source}: {message.content[:50]}...")
                return "\n\n".join(result_parts) if result_parts else "Анализ завершен, но агенты не дали ответа."

            result = loop.run_until_complete(analyze())
        finally:
            loop.close()
        logger.info(f"✅ Синхронный анализ завершен. Длина ответа: {len(result)} символов")
        return result
    except Exception as e:
        logger.error(f"❌ Ошибка в run_analysis_sync: {e}", exc_info=True)
        return f"Ошибка при анализе: {str(e)}"

# ==================== МАРШРУТЫ ====================
@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "CaptainAgent running", "endpoints": {"/": "GET", "/health": "GET", "/analyze": "POST"}}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    """Синхронный обработчик POST-запросов на анализ."""
    logger.info("📥 POST /analyze called")
    # Проверяем инициализацию
    if not model_client or not agent_team:
        logger.error("❌ Система не инициализирована")
        return jsonify({"error": "System not initialized"}), 503
    try:
        data = request.get_json()
        if not data:
            logger.warning("⚠️ Нет JSON в запросе")
            return jsonify({"error": "No JSON data"}), 400
        task = data.get('task')
        if not task:
            logger.warning("⚠️ Нет поля 'task' в JSON")
            return jsonify({"error": "Missing 'task' field"}), 400
        logger.info(f"📝 Получена задача: {task[:100]}...")
        # Запускаем анализ в синхронной обёртке
        result = run_analysis_sync(task)
        logger.info(f"✅ Отправляю результат клиенту")
        return jsonify({"result": result}), 200
    except Exception as e:
        logger.error(f"❌ Необработанная ошибка в /analyze: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# ==================== ЗАПУСК ====================
if __name__ == '__main__':
    if init_model_client() and init_agent_team():
        port = int(os.environ.get('PORT', 10000))
        logger.info(f"🚀 Запуск CaptainAgent (встроенный сервер) на порту {port}")
        logger.warning("⚠️ Используется встроенный сервер Flask. Для продакшена используйте Gunicorn.")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("❌ Не удалось инициализировать систему")
