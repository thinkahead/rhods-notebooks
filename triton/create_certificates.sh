# a script for creation of mutual TLS certificates

server_ip_address=9.42.68.239
server_dns=tritonserver.huggingface.svc.cluster.local

# Generate valid CA
openssl genrsa -passout pass:testtritoncapass -des3 -out ca.key 4096
openssl req -passin pass:testtritoncapass -new -x509 -days 365 -key ca.key -out ca.crt -subj  "/C=US/ST=New York/L=Brooklyn/O=Example/OU=Research/CN=Root CA"

# Generate valid Server Key/Cert
openssl genrsa -passout pass:testtritonserverpass -des3 -out server.key 4096
openssl req -passin pass:testtritonserverpass -new -key server.key -out server.csr -subj  "/C=US/ST=New York/L=Brooklyn/O=Example/OU=Server/CN=$server_ip_address"
echo subjectAltName = DNS:$server_dns,DNS:tritonserver.huggingface.svc,IP:$server_ip_address,IP:127.0.0.1 >> extfile.cnf
echo extendedKeyUsage = serverAuth >> extfile.cnf
openssl x509 -req -passin pass:testtritoncapass -days 365 -in server.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out server.crt -extfile extfile.cnf

# Remove passphrase from the Server Key
openssl rsa -passin pass:testtritonserverpass -in server.key -out server.key

# Generate valid Client Key/Cert
openssl genrsa -passout pass:testtritonclientpass -des3 -out client.key 4096
openssl req -passin pass:testtritonclientpass -new -key client.key -out client.csr -subj  "/C=US/ST=New York/L=Brooklyn/O=Example/OU=Client/CN=client"
openssl x509 -passin pass:testtritoncapass -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out client.crt

# Remove passphrase from Client Key
openssl rsa -passin pass:testtritonclientpass -in client.key -out client.key

rm extfile.cnf server.csr client.csr
