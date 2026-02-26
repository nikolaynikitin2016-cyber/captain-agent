import os
import json
import logging
from flask import Flask, request, jsonify

# Импорты для новой версии pyautogen 0.10.0
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models import OpenAIChatCompletionClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Загружаем библиотеку экспертов
with open('agent_library.json', 'r', encoding='utf-8') as f:
    agent_library = json.load(f)

# Настройка клиента для DeepSeek
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'task' not in data:
        return jsonify({'error': 'Missing task field'}), 400
    
    task = data['task']
    logger.info(f'Received task: {task}')
    
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
        
        # Запускаем анализ
        result = []
        async def run_analysis():
            async for message in team.run_stream(task=task):
                result.append(str(message))
        
        import asyncio
        asyncio.run(run_analysis())
        
        summary = "\n".join(result)
        logger.info('Analysis completed')
        return jsonify({'result': summary})
        
    except Exception as e:
        logger.exception('Error during analysis')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
