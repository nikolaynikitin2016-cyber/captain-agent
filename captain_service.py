import os
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
            system_message="Ты — технический аналитик. Анализируй графики и индикаторы. Отвечай кратко."
        )
        
        news_analyst = AssistantAgent(
            name="News_Analyst",
            model_client=model_client,
            system_message="Ты — новостной аналитик. Оценивай рыночные настроения."
        )
        
        decision_maker = AssistantAgent(
            name="Decision_Maker",
            model_client=model_client,
            system_message="Ты — главный аналитик. Собери отчеты и дай итоговую рекомендацию."
        )
        
        agent_team = RoundRobinGroupChat(
            participants=[tech_analyst, news_analyst, decision_maker],
            max_turns=3  # Уменьшил для скорости
        )
        logger.info("✅ Команда агентов создана")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка создания команды: {e}", exc_info=True)
        return False

# ==================== ИНИЦИАЛИЗАЦИЯ ПРИ ЗАПУСКЕ ====================
# ЭТО ВАЖНО: вызываем функции инициализации при импорте модуля
if not model_client:
    init_model_client()
if not agent_team and model_client:
    init_agent_team()
# ===================================================================

# ==================== МАРШРУТЫ ====================

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "CaptainAgent running",
        "version": "2.0",
        "endpoints": {
            "/": "GET - информация",
            "/health": "GET - проверка",
            "/analyze": "POST - анализ"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Проверка здоровья"""
    if model_client and agent_team:
        return jsonify({"status": "healthy", "agents": "ready"}), 200
    else:
        return jsonify({"status": "degraded", "agents": "initializing"}), 503

@app.route('/analyze', methods=['POST'])
def analyze():
    """Анализ задачи"""
    logger.info("📥 POST /analyze called")
    
    # Проверяем инициализацию
    if not model_client or not agent_team:
        logger.error("❌ Система не инициализирована")
        return jsonify({"error": "System not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data or 'task' not in data:
            return jsonify({"error": "Missing task"}), 400
        
        task = data['task']
        logger.info(f"📝 Задача: {task[:100]}...")
        
        # Запускаем анализ
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Здесь будет реальный анализ, пока заглушка
            result = f"✅ Анализ по запросу: '{task}'\n\n(Функция агентов временно отключена для теста)"
        finally:
            loop.close()
        
        return jsonify({"result": result}), 200
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Для локального запуска
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
