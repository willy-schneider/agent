FROM ollama/ollama:latest

# Копируем скрипт инициализации
COPY pull-models.sh /pull-models.sh
RUN chmod +x /pull-models.sh

# Устанавливаем curl для проверки готовности сервера (в образе его нет)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Запускаем скрипт
ENTRYPOINT ["/pull-models.sh"]
