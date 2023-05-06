```
oc new-project huggingface
```

Create the secret from the certificates
```
kubectl create secret generic triton-certificates --from-file=ca.crt --from-file=ca.key --from-file=client.crt --from-file=client.key --from-file=server.crt --from-file=server.key -n huggingface
```

Create a new image in OpenShift
```
oc -n huggingface new-build --name custom-triton-server --code https://github.com/thinkahead/rhods-notebooks --context-dir triton
watch oc get bc,build,is,pods -n huggingface
oc logs build/custom-triton-server-1 -n huggingface -f
oc wait --for=condition=complete build.build.openshift.io/custom-triton-server-1 -n huggingface --timeout=600s
oc delete bc -n huggingface --selector build=custom-triton-server
oc delete is -n huggingface --selector build=custom-triton-server
```
Use the image after it is built image-registry.openshift-image-registry.svc:5000/huggingface/custom-triton-server:latest

