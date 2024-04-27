import hvac
import os

# Authentication
client = hvac.Client(
    url='http://127.0.0.1:8200',
    token='sometesttoken',
)

secret1 = os.getenv('SECRET1')
secret2 = os.getenv('SECRET2')

# Writing a secret
create_response = client.secrets.kv.v2.create_or_update_secret(
    path='hdfs-secret',
    secret=dict(part1=secret1, part2=secret2))

print('Secret written successfully.')

