import os
import json
import logging
import sys
from flask import Flask, request, jsonify

# Настройка базового логирования ДО всех импортов, чтобы видеть ошибки сразу
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.info("🚀 Starting CaptainAgent service...")

# Попытка импорта с обработкой ошибок
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core import CancellationToken
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    logger.info("✅ All autogen modules imported successfully.")
except ImportError as e:
    logger.error(f"❌ Failed to import autogen modules: {e}")
    # Выходим с кодом 1, чтобы Render увидел ошибку и деплой провалился, показав лог
    sys.exit(1)

app = Flask(__name__)

# Загружаем библиотеку экспертов
try:
    with open('agent_library.json', 'r', encoding='utf-8') as f:
        agent_library = json.load(f)
    logger.info(f"✅ Loaded agent library with {len(agent_library)} experts.")
except FileNotFoundError:
    logger.error("❌ agent_library.json not found!")
    sys.exit(1)
except json.JSONDecodeError as e:
    logger.error(f"❌ Error decoding agent_library.json: {e}")
    sys.exit(1)

# Настройка клиента для DeepSeek
try:
    model_client = OpenAIChatCompletionClient(
        model=os.getenv('LLM_MODEL', 'deepseek-chat'),
        api_key=os.getenv('LLM_API_KEY'),
        base_url=os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1'),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "deepseek"
        }
    )
    logger.info("✅ Model client configured successfully.")
except Exception as e:
    logger.error(f"❌ Failed to configure model client: {e}")
    sys.exit(1)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'task' not in data:
        return jsonify({'error': 'Missing task field'}), 400

    task = data['task']
    logger.info(f'📥 Received task: {task}')

    try:
        # Создаем агентов из библиотеки
        agents = []
        for expert in agent_library[:3]:  # Берем первых 3 экспертов для простоты
            agent = AssistantAgent(
                name=expert['name'].replace(' ', '_'),
                description=expert['description'],
                model_client=model_client,
                system_message=expert['system_message']
            )
            agents.append(agent)

        # Создаем команду
        team = RoundRobinGroupChat(agents)
        logger.info(f"✅ Created team with {len(agents)} agents.")

        # Запускаем анализ
        result = []
        async def run_analysis():
            async for message in team.run_stream(task=task):
                result.append(str(message))

        import asyncio
        asyncio.run(run_analysis())

        summary = "\n".join(result)
        logger.info('✅ Analysis completed')
        return jsonify({'result': summary})

    except Exception as e:
        logger.exception('❌ Error during analysis')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # ВАЖНО: Render ожидает порт 10000 по умолчанию
    port = int(os.getenv('PORT', 10000))
    logger.info(f"✅ Starting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port)
