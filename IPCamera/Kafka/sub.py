from kafka import KafkaConsumer
import json

# Tema al que se suscribirá el consumidor
topic_name = 'mi_tema_kafka'

# Configuración del consumidor
# bootstrap_servers: lista de direcciones de brokers de Kafka
# group_id: ID del grupo de consumidores. Si múltiples consumidores tienen el mismo group_id
#           Kafka distribuirá las particiones del tema entre ellos.
# auto_offset_reset: 'earliest' para empezar a leer desde el inicio del tema
#                    'latest' para empezar a leer solo los mensajes nuevos
# enable_auto_commit: True para que el consumidor haga commit automático de los offsets
# value_deserializer: función para deserializar los bytes del mensaje a un formato utilizable
consumer = KafkaConsumer(
    topic_name,
    bootstrap_servers=['localhost:9092'],
    group_id='mi_grupo_consumidor', # Puedes usar un ID de grupo diferente si quieres consumidores independientes
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print(f"Iniciando consumidor para el tema: {topic_name} en el grupo: mi_grupo_consumidor")
print("Esperando mensajes...")

try:
    for message in consumer:
        # message es un objeto ConsumerRecord que contiene información del mensaje
        # message.value: el valor del mensaje (ya deserializado)
        # message.topic: el tema del mensaje
        # message.partition: la partición del tema de donde vino el mensaje
        # message.offset: el offset del mensaje dentro de la partición
        print(f"Mensaje recibido:")
        print(f"  - Topic: {message.topic}")
        print(f"  - Partition: {message.partition}")
        print(f"  - Offset: {message.offset}")
        print(f"  - Value: {message.value}")
        
        # Aquí puedes agregar la lógica para procesar el mensaje
        # Por ejemplo, guardarlo en una base de datos, enviarlo a otro servicio, etc.

except KeyboardInterrupt:
    print("\nConsumidor detenido manualmente.")
finally:
    consumer.close()
    print("Consumidor cerrado.")