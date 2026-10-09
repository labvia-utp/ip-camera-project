from kafka import KafkaProducer
import json
import time

# Configuración del productor
# bootstrap_servers: lista de direcciones de brokers de Kafka
# value_serializer: función para serializar el valor del mensaje a bytes
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Tema al que se enviarán los mensajes
topic_name = 'mi_tema_kafka'

print(f"Iniciando productor para el tema: {topic_name}")

for i in range(10):
    message_data = {
        'id': i,
        'mensaje': f'Hola desde Python, mensaje número {i}',
        'timestamp': time.time()
    }
    try:
        # Envía el mensaje al tema
        # .send() es asíncrono y devuelve un objeto Future
        future = producer.send(topic_name, value=message_data)
        
        # Bloquea hasta que el mensaje sea enviado (opcional, para envíos síncronos)
        record_metadata = future.get(timeout=10) 
        
        print(f"Mensaje enviado exitosamente: {message_data}")
        print(f"  - Topic: {record_metadata.topic}")
        print(f"  - Partition: {record_metadata.partition}")
        print(f"  - Offset: {record_metadata.offset}")
        
    except Exception as e:
        print(f"Error al enviar el mensaje {message_data}: {e}")
    
    time.sleep(1) # Espera 1 segundo antes de enviar el siguiente mensaje

# Asegura que todos los mensajes pendientes se envíen antes de cerrar
producer.flush()
print("Productor finalizado.")