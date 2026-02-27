import os
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "CaptainAgent running (test mode)"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Простой тестовый обработчик, который всегда отвечает."""
    logger.info("📥 POST /analyze called")
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON"}), 400
        task = data.get('task', '')
        logger.info(f"📝 Received task: {task[:50]}...")
        # Простой ответ без вызова Autogen
        return jsonify({"result": f"✅ Получен запрос: '{task}'. Это тестовый ответ. CaptainAgent в режиме отладки."}), 200
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
