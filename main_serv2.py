import os
import asyncio
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import pandas as pd
import numpy as np
import torch
import nest_asyncio
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from peft import PeftConfig, PeftModel
from transformers import (
     AutoModelForCausalLM,
     AutoTokenizer,
     pipeline,
     GPTQConfig
 )
from googletrans import Translator
nest_asyncio.apply()
storage = MemoryStorage()
class UserStates(StatesGroup):
  start_bot = State()
  to_questionary = State()
  to_analysis = State()
  after_analysis = State()
  save_file = State()
  to_process_analysis = State()
  to_AI = State()
token = '6330602631:AAGD-y1wboKQSXOkyJUWlV7UVXmNRVgPN90'
bot = Bot(token)
dp = Dispatcher(bot, storage=storage)
MAX_QUESTIONS = 33
# начало нейронки

translator = Translator()
torch.cuda.is_available()

### Инициализация модели ######################################################################################
# Эта ячейка загружает модель в видеокарту, ни в коем случае ее нельзя запускать несколько раз за одну сессию,
# иначе в видюху загрузится несколько моделей и в ней начнет закачниватся видеопамять,
# если все таки нужно повторно запустить код, то сначала перезапустите среду чтобы освободить
# память от прошлой модели.
# По умолчанию модель занимает не более 10 ГБ VRAM, если занято больше значит вы что то делаете не так
#####################################################################################################################

print(torch.cuda.is_available())
base_model_name = "TheBloke/Llama-2-13B-chat-GPTQ"
adapter_model = '/home/ubuntu/model'

print('model init start')
model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                              device_map={"": 0},
                                              revision = 'gptq-4bit-32g-actorder_True',
                                              quantization_config=GPTQConfig(bits=4,use_exllama=True))
model = PeftModel.from_pretrained(model, adapter_model)
print('tokenizer init start')
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

### Параметры модели ###

pipe = pipeline('text-generation',model=model,tokenizer=tokenizer,
                 max_new_tokens=300, # Максимальная длина ответа, сильное снижение может привести к обрезанию ответов
                 do_sample=True,   # не трогать
                 temperature=0.1,  # рандомность ответа, чем больше это число тем более неожиданные и креативные ответы будет давать модель, если число маленькое то ответы будут однообразными но уверенными.
                 top_p=0.95, #  тоже рандомность, чем меньше тем больше модель будет отходить от контекста.
                 top_k=40, # аналагично параметру выше, не рекомендуется менять.
                 repetition_penalty=1.15  # штраф за повторение, чем выше тем меньше модель будет повторять одни и те же слова/предложения, слишком высокое значение может привести к бреду/галлюцинациям
                )

# # ### Системный промпт, в нем вы обьясняете боту кто он по жизни и в чем его цель, что ему стоит делать а что нет, можно свободно изменять.
system_prompt = "You are helpful assistant-nutritionist that answers patient's questions. Answer as short as possible."

print('model init done')

#конец нейронки



# Берем результаты опросника и на основе них выводим возможные дефициты:



# Получаем информацию о результатах опроса конкретного пользователя
async def get_user_data(user_id, db, specific_id):
  hypothyroidism = db['Г'].to_list()[specific_id]
  insulinresistance = db['ИР'].to_list()[specific_id]
  irondeficit = db['ЖД'].to_list()[specific_id]
  return hypothyroidism,insulinresistance,irondeficit


# Получаем id пользователя в базе пользователей
async def get_user_id_in_base(user_id):
  with open('user_data.csv') as f:
    db = pd.read_csv('user_data.csv')
    user_ids = list(db['user_id'])
    if user_id in user_ids:
      specific_id = user_ids.index(user_id)
    else:
      specific_id = None
  return specific_id, db

async def print_instructions(message: types.Message, markup=None):
  await bot.send_message(message.chat.id, "Подготовка к анализам:\n"
                      "Общие правила подготовки и сдачи анализов крови на витамины и минералы:\n"
                      "●прийти в лабораторию заранее, чтобы привести в порядок эмоциональное и физическое состояние\n"
                      "●с момента последнего приема пищи должно пройти не менее восьми часов\n"
                      "●с утра разрешается пить воду, только чистую без добавок\n"
                      "●за неделю до сдачи (примерно) отказаться от употребления содержащих спирт напитков (в Т.Ч.\n"
                      "аптечной продукции)\n"
                      "●в день сдачи крови желательно воздерживаться от курения\n"
                      "●нельзя совмещать дни посещения лаборатории и физиотерапевтических (аппаратных) процедур исключить интенсивные физические нагрузки\n"
                      "Если Вы находитесь в состоянии сильных эмоциональных переживаний, возможно получение искаженных результатов.")


# Подгружаем все необходимые файлы
user_agreement = open(r'Пользовательское соглашение.pdf', 'rb')  # название

process_data = open(r'Обработка_персональных_данных.pdf', 'rb')

questions = open(r'questions.txt', 'rb')

zinc = open(r'Цинк.pdf', 'rb')

chrome = open(r'Хром.pdf', 'rb')

acid = open(r'Фолиевая кислота.pdf', 'rb')

test_results = open(r"xlsx/Показания.xlsx", 'rb')

selen = open(r'Селен.pdf', 'rb')

copper = open(r'Медь.pdf', 'rb')

magnium = open(r'Магний.pdf', 'rb')

b12 = open(r'Кобаламин (B12).pdf', 'rb')

iodine = open(r'Йод.pdf', 'rb')

fats = open(r'Жиры.pdf', 'rb')

iron = open(r'Железо.pdf', 'rb')

vitamineC = open(r'Витамин C.pdf', 'rb')

#file_id = find_file()
db = open(r'user_data.csv', 'rb')

# Обработку опросника необходимо было вынести в отдельную асинхронную функцию
async def process_quest(message: types.Message, question_num):
  specific_id, db = await get_user_id_in_base(message.from_user.id)

  hypothyroidism, insulinresistance, irondeficit = await get_user_data(message.from_user.id, db, specific_id)
  if question_num < 33:
    if message.text == "Да":
      if question_num == 2:
        irondeficit += 1
      if question_num < 9:
        hypothyroidism += 1
      elif 9 <= question_num < 21:
        insulinresistance += 1
      elif 22 <= question_num < 34:
        irondeficit += 1
      question_num = question_num + 1
    else:
      question_num = question_num + 1
  return question_num, hypothyroidism, insulinresistance, irondeficit



# @bot.message_handler(content_types=['text', 'audio', 'document', 'animation', 'game', 'photo', 'sticker', 'video', 'video_note', 'voice', 'location', 'contact', 'venue', 'dice', 'new_chat_members', 'left_chat_member', 'new_chat_title',
#                   'new_chat_photo', 'delete_chat_photo', 'group_chat_created', 'supergroup_chat_created', 'channel_chat_created', 'migrate_to_chat_id', 'migrate_from_chat_id', 'pinned_message', 'invoice', 'successful_payment',
#                   'connected_website', 'poll', 'passport_data', 'proximity_alert_triggered', 'video_chat_scheduled', 'video_chat_started', 'video_chat_ended', 'video_chat_participants_invited', 'web_app_data',
#                   'message_auto_delete_timer_changed', 'forum_topic_created', 'forum_topic_closed', 'forum_topic_reopened', 'forum_topic_edited', 'general_forum_topic_hidden', 'general_forum_topic_unhidden', 'write_access_allowed',
#                  'user_shared', 'chat_shared', 'story'])

@dp.message_handler()
async def sample(message):
    global download_db, file_id
    questions = list(open('questions.txt', encoding='utf-8'))
    # Блок с приветствием от бота
    if message.content_type == 'text' and (message.text.lower() == 'старт' or message.text.lower() == '/start'):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        but1 = types.KeyboardButton('✅ Начать')
        markup.add(but1)
        await bot.send_message(message.chat.id,
                         "Здравствуйте! Я Онлайн нутрицолог, моя задача - помочь Вам"
                         " с решением возможных проблем со здоровьем путём применения"
                         " <b>БАДов и витаминов</b>. Приступим к работе?", parse_mode="html", reply_markup=markup)
        await UserStates.start_bot.set()
    else: await bot.send_message(message.chat.id,"Пожалуйста, отвечайте корректно. Используйте кнопки, предложенные под полем для ответа.")



@dp.message_handler(state=UserStates.start_bot)
async def start_bot(message: types.Message, state: FSMContext):

    # Блок с пользовательским соглашением
    if message.content_type == 'text' and message.text == '✅ Начать':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        but2 = types.KeyboardButton('Подтверждаю')
        markup.row(but2)
        await bot.send_document(message.chat.id, open(r'Пользовательское соглашение.pdf', 'rb'))
        await bot.send_document(message.chat.id, open(r'Обработка_персональных_данных.pdf', 'rb'))
        await bot.send_message(message.chat.id, 'ℹ️ Прежде чем работать с ботом, <b>обязательно</b>'
                                            ' обратите внимание на пользовательское соглашение.'
                                            '\n \nНажимая кнопку "Подтверждаю", Вы автоматически считаетесь '
                                            'ознакомленными'
                                            ' и согласными с его условиями.', parse_mode='html', reply_markup=markup)


    # Блок со службой поддержки
    elif message.content_type == 'text' and message.text == 'Подтверждаю':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        button4 = types.KeyboardButton('✅ Хорошо')
        markup.row(button4)
        await bot.send_message(message.chat.id, "ℹ️ Обратите внимание, что у нас есть <b>Служба Поддержки</b>,"
                                            " в которую Вы можете обратиться в случае возникновения каких-либо жалоб"
                                            " или вопросов."
                                            "\n<b>Электронная почта Службы Поддержки </b>: lunarfly_off@mail.ru"
                                            "\nСреднее время ответа <b>Службы Поддержки </b>занимает от 1 до 3 рабочих дней.",
                          parse_mode="html", reply_markup=markup)
        await UserStates.to_questionary.set()
    else:
        await bot.send_message(message.chat.id,"Пожалуйста, отвечайте корректно. Используйте кнопки, предложенные под полем для ответа.")

@dp.message_handler(state=UserStates.to_questionary)
async def start_questionary(message: types.Message,state: FSMContext):
    questions = list(open('questions.txt', encoding='utf-8'))
    # !!!Пишите все, что должно быть после соглашения в блоке снизу
    if message.content_type == 'text' and (message.text == '✅ Хорошо' or message.text == "Начать проходить заново"):
      specific_id, db = await get_user_id_in_base(message.from_user.id)
      if specific_id != None:
        db = db.drop(db.index[specific_id])
        db.to_csv("user_data.csv", index=False)
      with open('user_data.csv', 'a') as f:
        f.write(f"{message.from_user.id}, {1}, {0}, {0}, {0}, {0}\n") # 1 1 0 0 0 -> 1 0 0 0 0
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
      button3 = types.KeyboardButton("Приступаем")
      markup.row(button3)
      await bot.send_message(message.chat.id, "Итак, начнем с опросника. В нём будут называться различные симптомы, которые могут быть представлены у Вас."
                                        " Отвечайте да, если сталкиваетесь с их проявлением, либо нет, если симптом Вас не беспокоит."
                                        " Просим отвечать честно, только так мы сможем дать Вам рекомендации.",
                        reply_markup=markup)


    # Начало блока с опросником
    elif message.content_type == 'text' and message.text == 'Приступаем':
      specific_id, db = await get_user_id_in_base(message.from_user.id)
      question_num = list(db['question'])[specific_id]
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
      buttonYes = types.KeyboardButton('Да')
      buttonNo = types.KeyboardButton('Нет')
      buttonStartAgain = types.KeyboardButton("Начать проходить заново")
      markup.row(buttonYes, buttonNo, buttonStartAgain)
      await bot.send_message(message.chat.id, f"{questions[question_num]}",reply_markup=markup)

    # Открываем БД с данными о пользователе
    elif message.content_type == 'text' and (message.text == "Да" or message.text == "Нет"):
      specific_id, db = await get_user_id_in_base(message.from_user.id)
      question_num = list(db['question'])[specific_id]
      question_num, hypothyroidism, insulinresistance, irondeficit = await process_quest(message, question_num)

        # На последнем вопросе уходим сюда, и только тогда он его засчитает
      if question_num == 33:
        db = db.drop(db.index[specific_id])
        db.to_csv("user_data.csv", index=False)
        with open('user_data.csv', 'a') as f:
          f.write(f"{message.from_user.id}, {1}, {question_num}, {hypothyroidism}, {insulinresistance}, {irondeficit}\n")
          db = pd.read_csv('user_data.csv')
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        agreeButton = types.KeyboardButton('Узнать результаты')
        markup.row(agreeButton)
        await bot.send_message(message.chat.id, f"{questions[question_num]}",reply_markup=markup)
      else:
        db = db.drop(db.index[specific_id])
        db.to_csv("user_data.csv", index=False)
        with open('user_data.csv', 'a') as f:
            f.write(f"{message.from_user.id}, {1}, {question_num}, {hypothyroidism}, {insulinresistance}, {irondeficit}\n")
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
        buttonYes = types.KeyboardButton('Да')
        buttonNo = types.KeyboardButton('Нет')
        buttonStartAgain = types.KeyboardButton("Начать проходить заново")
        markup.row(buttonYes, buttonNo, buttonStartAgain)
        await bot.send_message(message.chat.id, f"{questions[question_num]}",reply_markup=markup)

    elif message.content_type == 'text' and message.text == "Узнать результаты":
      specific_id, db = await get_user_id_in_base(message.from_user.id)
      hypothyroidism, insulinresistance, irondeficit = await  get_user_data(message.from_user.id, db, specific_id)
      await show_results(message, hypothyroidism, insulinresistance, irondeficit, specific_id, db, message.from_user.id)

    elif message.text == "Начать проходить заново":
      await UserStates.to_questionary.set()
    else:
      await bot.send_message(message.chat.id, "Пожалуйста, отвечайте корректно. Используйте кнопки, предложенные под полем для ответа.")

@dp.message_handler()
async def show_results(message: types.Message, hypothyroidism: int, insulinresistance: int, irondeficit: int, specific_id: int, db, user_id):
    if hypothyroidism == irondeficit == insulinresistance == 0:
        await bot.send_message(message.chat.id, "Вы можете сдать следующие анализы(даны в табличке ниже), чтобы быть уверенными, что никаких проблем у Вас нет."
                                              "Результаты анализов впишите в данную табличку и отправьте боту")
        await print_instructions(message)
        await bot.send_document(message.chat.id, open('xlsx/analysis3.xlsx', 'rb'))
        await UserStates.to_analysis.set()
      #  dp.message.register(get_analysis, UserState to_analysis)
    else:
      db = db.drop(db.index[specific_id])
      db.to_csv("user_data.csv", index=False)
      with open('user_data.csv', 'a') as f:
        markup = types.ReplyKeyboardMarkup(resize_keyboard = True, one_time_keyboard = True)
        okButton = types.KeyboardButton("Понятно")
        markup.row(okButton)
        await bot.send_message(message.chat.id, "У Вас возможны дефициты следующих элементов: ")
        if hypothyroidism > irondeficit or hypothyroidism > insulinresistance:
            await bot.send_document(message.chat.id, open(r"Цинк.pdf", 'rb'))
            await bot.send_document(message.chat.id, open(r"Йод.pdf", 'rb'))
            await bot.send_document(message.chat.id, open(r"Селен.pdf", 'rb'))
            await bot.send_document(message.chat.id, open(r"Жиры.pdf", 'rb'), reply_markup=markup)
            f.write(f"{user_id}, {1}, {MAX_QUESTIONS}, {hypothyroidism}, {insulinresistance}, {irondeficit}, {1}\n")
            await UserStates.after_analysis.set()
        elif irondeficit > insulinresistance or irondeficit >= hypothyroidism:
            await bot.send_document(message.chat.id, open(r"Цинк.pdf", 'rb'))
            await bot.send_document(message.chat.id, open(r"Хром.pdf", 'rb'))
            await bot.send_document(message.chat.id, open(r"Магний.pdf", 'rb'), reply_markup=markup)
            f.write(f"{user_id}, {1}, {MAX_QUESTIONS}, {hypothyroidism}, {insulinresistance}, {irondeficit}, {2}\n")
            await UserStates.after_analysis.set()
        elif irondeficit == insulinresistance or insulinresistance > irondeficit or insulinresistance >= hypothyroidism:
            await bot.send_document(message.chat.id, open(r"Кобаламин (B12).pdf", 'rb'))
            await bot.send_document(message.chat.id, open(r"Фолиевая кислота.pdf", 'rb'))
            await bot.send_document(message.chat.id, open(r"Медь.pdf", 'rb'))
            await bot.send_document(message.chat.id, open(r"Витамин C.pdf", 'rb'))
            await bot.send_document(message.chat.id, open(r"Железо.pdf", 'rb'), reply_markup = markup)
            f.write(f"{user_id}, {1}, {MAX_QUESTIONS}, {hypothyroidism}, {insulinresistance}, {irondeficit}, {3}\n")
            await UserStates.after_analysis.set()

@dp.message_handler(state=UserStates.after_analysis)
async def start_after_analysis(message: types.Message, state: FSMContext):
    if message.content_type == 'text' and message.text == "Понятно":
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard = True)
      continueButton1 = types.KeyboardButton("Продолжить")
      markup.row(continueButton1)
      await bot.send_message(message.chat.id, "Если Вы хотите узнать, какой комплекс БАДов Вам необходим,\n"
                                        "нужно сдать анализы. Продолжаем?", reply_markup = markup)


    elif message.content_type == 'text' and message.text == 'Продолжить':
      markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard = True)
      continueButton2 = types.KeyboardButton("Прочитано")
      markup.row(continueButton2)
      await print_instructions(message)
      await bot.send_message(message.chat.id, "Пожалуйста, обращайте внимание на единицы измерений, которые указаны в таблице."
        "В случае, когда Вы записываете десятичное число через точку и таблица меняет его на дату, запишите это число"
        "через запятую. При записи чисел НЕ используйте пробелы(при указании диапазона используйте дефис).", reply_markup=markup)

    # Выдаем соответствующую табличку, куда записывать результаты
    elif (message.content_type == 'text' or message.content_type == 'document') and message.text == "Прочитано":
      specific_id, db = await  get_user_id_in_base(message.from_user.id)
      table_num = list(db['table_num'])[specific_id]
      await bot.send_message(message.chat.id, "В данной ниже таблице вместо нулей заполните Ваши результаты анализов."
                       "Удостоверьтесь, что Ферритин, ТТГ и инсулин заполнены!")
      if table_num == 1:
        await bot.send_document(message.chat.id, open('xlsx/analysis3.xlsx', 'rb'))
        await UserStates.to_analysis.set()
      elif table_num == 2:
        await bot.send_document(message.chat.id, open('xlsx/analysis1.xlsx', 'rb'))
        await UserStates.to_analysis.set()
      else:
        await bot.send_document(message.chat.id, open('xlsx/analysis2.xlsx', 'rb'))
        await UserStates.to_analysis.set()

    else:
      await bot.send_message(message.chat.id, "Пожалуйста, отвечайте корректно. Используйте кнопки, предложенные под полем для ответа.")
# Получаем анализы от пользователя
@dp.message_handler(content_types=types.ContentType.ANY, state=UserStates.to_analysis)
async def get_analysis(message: types.Message, state: FSMContext):
  try:
    file_name = message.document.file_name
    result_table = await bot.get_file(message.document.file_id)
    downloaded_file = await bot.download_file(result_table.file_path)
    with open(file_name, 'wb') as new_file:
      new_file.write(downloaded_file.getvalue())
    async with state.proxy() as data:
      data['table'] = file_name
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard = True)
    manButton = types.KeyboardButton("Мужчина")
    womanButton = types.KeyboardButton("Женщина")
    markup.row(manButton, womanButton)
    await bot.send_message(message.chat.id, "Для корректных результатов, необходимо узнать Ваш пол.", reply_markup = markup)
    await UserStates.to_process_analysis.set()#Поменять это и все остальные next_step_handler

  except Exception:
    await bot.send_message(message.chat.id, "Упс, произошла ошибка, попробуйте еще раз!")
    await UserStates.to_analysis.set()#Поменять это и все остальные next_step_handler

@dp.message_handler(state=UserStates.to_process_analysis)
async def process_analysis(message: types.Message, state: FSMContext):
  print(message.text)
  if message.text != "Мужчина" and message.text != "Женщина":
    await bot.send_message(message.chat.id, "Некорректный пол.  Отправьте заново таблицу.")
    await UserStates.to_analysis.set()
  else:
      print(message.from_user.id)
      specific_id, db = await get_user_id_in_base(message.from_user.id)
      print(specific_id)
      hypothyroidism, insulinresistance, irondeficit = await get_user_data(message.from_user.id, db, specific_id)
      db = db.drop(db.index[specific_id])
      db.to_csv("user_data.csv", index=False)
      async with state.proxy() as data:
        file_name=data['table']
      results_to_compare = pd.DataFrame(pd.read_excel(file_name))
      with open('user_data.csv', 'a') as f:
        f.write(f"{message.from_user.id}, {1}, {MAX_QUESTIONS}, {hypothyroidism}, {insulinresistance}, {irondeficit}, {0}, {message.text}\n")
        db = pd.read_csv('user_data.csv')
        os.remove(file_name)
      main_analysis = {}
      for i in range(results_to_compare.count()[0]):
        row = list(results_to_compare.loc[i])
        if row[1] != 0 and not(pd.isnull(row[1])):
          if row[0] == "Инсулин" or row[0] == "ТТГ" or row[0] == "Ферритин":
            if row[1].is_integer() or isinstance(row[1], np.floating):
              main_analysis[row[0]] = row[1]


      # Саша, здесь есть повторяющиеся строки. Если у тебя есть идея, как от них избавиться, то давай :))) /// в ф-ию var_sender вынес все эти операции (она выше в комменнтариях).
      if ("Инсулин" not in main_analysis) or ("Ферритин" not in main_analysis) or ("ТТГ" not in main_analysis):
        await bot.send_message(message.chat.id, "Вы ввели некорректные данные в таблице, попробуйте еще раз!"
                         "Удостоверьтесь, что Ферритин, ТТГ и инсулин заполнены!")
        await UserStates.to_analysis.set()
      elif 50 <= main_analysis["Ферритин"] <= 150 and 2 <= main_analysis["Инсулин"] <= 6 and 0.4 <= float(main_analysis["ТТГ"]) <= 2.0:
        await bot.send_document(message.chat.id, open(r"Вариант4.pdf", 'rb'))
        await bot.send_message(message.chat.id, "Если у вас остались вопросы, можете задать их в свободной форме ниже, на них Вам ответит искусственный интеллект[beta].\n"
        "Начинайте спрашивать прямо сейчас. Если у Вас нет вопросов или Вы захотите остановиться, напишите СТОП.")
        await UserStates.to_AI.set()
      else:
        if main_analysis["Ферритин"] <= main_analysis["Инсулин"] or main_analysis["Инсулин"] >= main_analysis["ТТГ"]:
          await bot.send_document(message.chat.id, open(r"Вариант1.pdf", 'rb'))
          await bot.send_message(message.chat.id, "Если у вас остались вопросы, можете задать их в свободной форме ниже, на них Вам ответит искусственный интеллект[beta].\n"
        "Начинайте спрашивать прямо сейчас. Если у Вас нет вопросов или Вы захотите остановиться, напишите СТОП.")
          await UserStates.to_AI.set()
        elif  main_analysis["Ферритин"] >= main_analysis["ТТГ"] or main_analysis["Ферритин"] > main_analysis["Инсулин"]:
          await bot.send_document(message.chat.id, open(r"Вариант2.pdf", 'rb'))
          await bot.send_message(message.chat.id, "Если у вас остались вопросы, можете задать их в свободной форме ниже, на них Вам ответит искусственный интеллект[beta].\n"
        "Начинайте спрашивать прямо сейчас. Если у Вас нет вопросов или Вы захотите остановиться, напишите СТОП.")
          await UserStates.to_AI.set()
        else:
          await bot.send_document(message.chat.id, open(r"Вариант3.pdf", 'rb'))
          await bot.send_message(message.chat.id, "Если у вас остались вопросы, можете задать их в свободной форме ниже, на них Вам ответит искусственный интеллект[beta].\n"
        "Начинайте спрашивать прямо сейчас. Если у Вас нет вопросов или Вы захотите остановиться, напишите СТОП.")
          await UserStates.to_AI.set()

@dp.message_handler(state=UserStates.to_AI)
async def ask_model(user_prompt: str, state: FSMContext):
  message = user_prompt
  if user_prompt.text.lower() != 'стоп':
    await bot.send_message(message.chat.id, 'Бот думает, подождите...')
    user_prompt = translator.translate(user_prompt.text).text
    output = pipe(f'''[INST] <<SYS>>{system_prompt}<</SYS>>{user_prompt}[/INST]''', return_full_text = False)[0]['generated_text']
    translated_output = translator.translate(output, src="en", dest="ru")
    await bot.send_message(message.chat.id, translated_output.text)
    await UserStates.to_AI.set()
  else:
    againButton = types.KeyboardButton("Начать проходить заново")
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard = True)
    markup.row(againButton)
    await bot.send_message(message.chat.id, "Вы можете начать проходить бота заново.", reply_markup = markup)
    await UserStates.to_questionary.set()



# Здесь запускается бот
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
