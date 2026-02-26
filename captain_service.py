import os
import json
import logging
from flask import Flask, request, jsonify
from autogen_agentchat import *
from autogen_core import *from autogen.agentchat.contrib.captainagent import CaptainAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Загружаем библиотеку экспертов
with open('agent_library.json', 'r', encoding='utf-8') as f:
    agent_library = json.load(f)

# Настройка LLM из переменных окружения
config_list = [
    {
        'model': os.getenv('LLM_MODEL', 'deepseek-chat'),
        'api_key': os.getenv('LLM_API_KEY'),
        'base_url': os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1'),
        'api_type': 'openai',
    }
]

llm_config = {
    'config_list': config_list,
    'temperature': 0.7,
    'timeout': 120,
}

# Создаём CaptainAgent (глобально)
captain = CaptainAgent(
    name='captain',
    llm_config=llm_config,
    agent_library=agent_library,
    code_execution_config={
        'use_docker': False,
        'work_dir': '/tmp/captain_work'
    },
    max_agents_per_task=5
)

user_proxy = autogen.UserProxyAgent(
    name='user_proxy',
    human_input_mode='NEVER',
    code_execution_config=False
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
        result = user_proxy.initiate_chat(
            captain,
            message=task,
            max_turns=5,
            clear_history=True
        )
        summary = result.summary if hasattr(result, 'summary') else str(result)
        logger.info('Analysis completed')
        return jsonify({'result': summary})
    except Exception as e:
        logger.exception('Error during analysis')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
