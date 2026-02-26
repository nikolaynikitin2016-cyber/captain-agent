import os
import json
import logging
import sys
from flask import Flask, request, jsonify

# Настройка базового логирования ДО всех импортов
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)
logger.info("🚀 Starting CaptainAgent service...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Files in directory: {os.listdir('.')}")

# Проверяем наличие agent_library.json
if not os.path.exists('agent_library.json'):
    logger.error("❌ agent_library.json NOT FOUND!")
    sys.exit(1)
else:
    logger.info("✅ agent_library.json found")

# Попытка импорта с обработкой ошибок
logger.info("Attempting to import autogen modules...")
try:
    from autogen_agentchat.agents import AssistantAgent
    logger.info("✅ Imported AssistantAgent")
    from autogen_agentchat.teams import RoundRobinGroupChat
    logger.info("✅ Imported RoundRobinGroupChat")
    from autogen_core import CancellationToken
    logger.info("✅ Imported CancellationToken")
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    logger.info("✅ Imported OpenAIChatCompletionClient")
    logger.info("✅ All autogen modules imported successfully.")
except ImportError as e:
    logger.error(f"❌ Failed to import autogen modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__)
logger.info("✅ Flask app created")

# Загружаем библиотеку экспертов
try:
    with open('agent_library.json', 'r', encoding='utf-8') as f:
        agent_library = json.load(f)
    logger.info(f"✅ Loaded agent library with {len(agent_library)} experts.")
    logger.info(f"First expert: {agent_library[0]['name'] if agent_library else 'None'}")
except FileNotFoundError:
    logger.error("❌ agent_library.json not found!")
    sys.exit(1)
except json.JSONDecodeError as e:
    logger.error(f"❌ Error decoding agent_library.json: {e}")
    sys.exit(1)

# Настройка клиента для DeepSeek
logger.info("Configuring model client...")
try:
    model = os.getenv('LLM_MODEL', 'deepseek-chat')
    api_key = os.getenv('LLM_API_KEY')
    base_url = os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1')
    
    logger.info(f"Model: {model}")
    logger.info(f"Base URL: {base_url}")
    logger.info(f"API Key present: {'Yes' if api_key else 'No'}")
    
    model_client = OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
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
    import traceback
    traceback.print_exc()
    sys.exit(1)

@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check called")
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
        for i, expert in enumerate(agent_library[:3]):
            logger.info(f"Creating agent {i+1}: {expert['name']}")
            agent = AssistantAgent(
                name=expert['name'].replace(' ', '_'),
                description=expert['description'],
                model_client=model_client,
                system_message=expert['system_message']
            )
            agents.append(agent)
        logger.info(f"✅ Created team with {len(agents)} agents.")

        # Создаем команду
        team = RoundRobinGroupChat(agents)
        logger.info("✅ Team created")

        # Запускаем анализ
        result = []
        async def run_analysis():
            logger.info("Starting analysis...")
            async for message in team.run_stream(task=task):
                result.append(str(message))
            logger.info("Analysis complete")

        import asyncio
        asyncio.run(run_analysis())

        summary = "\n".join(result)
        logger.info('✅ Analysis completed')
        return jsonify({'result': summary})

    except Exception as e:
        logger.exception('❌ Error during analysis')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    logger.info(f"✅ Starting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port)
