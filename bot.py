import os
import logging
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import ParseMode

# Настройки
TELEGRAM_TOKEN = "ВАШ_ТОКЕН_ОТ_BOTFATHER"
CAPTAIN_API_URL = "https://captain-agent.onrender.com/analyze"
ALLOWED_USERS = [ВАШ_TELEGRAM_ID]  # Узнайте у @userinfobot

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

def is_allowed(user_id):
    return user_id in ALLOWED_USERS

@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    if not is_allowed(message.from_user.id):
        await message.reply("⛔ Доступ запрещён")
        return
    await message.reply("🚀 CaptainAgent готов! Отправьте задачу, например:\n\n"
                       "`Проанализируй Bitcoin на сегодня`", parse_mode="Markdown")

@dp.message_handler()
async def handle_task(message: types.Message):
    if not is_allowed(message.from_user.id):
        await message.reply("⛔ Доступ запрещён")
        return

    task = message.text
    waiting_msg = await message.reply("⏳ Анализирую... (30-60 секунд)")

    try:
        response = requests.post(
            CAPTAIN_API_URL,
            json={'task': task},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        result = data.get('result', 'Нет результата')
    except Exception as e:
        result = f"❌ Ошибка: {str(e)}"

    await waiting_msg.edit_text(f"✅ Результат:\n\n{result[:4000]}", parse_mode=ParseMode.HTML)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
