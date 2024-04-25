from hdfs import InsecureClient

client = InsecureClient('http://127.0.0.1:9870', user='mlops')

#client.makedirs('/service_log', permission='766')

log=['hah1\n', 'lal2\n', 'kek3\n']

#client.write('/service_log/log', data=''.join(log[-2:]), overwrite=True, permission='766')
#client.write('/service_log/log', data=log[0], overwrite=True)
with client.read('/service_log/log', encoding='utf-8') as reader:
    content = reader.read()
    print(content)

client.write('/service_log/log', data=content+''.join(log[-2:]), overwrite=True)