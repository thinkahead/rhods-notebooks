Create a new image in OpenShift

```
oc -n huggingface new-build --name custom-mnist-image --code https://github.com/thinkahead/mnist-test
watch oc get bc,build,is,pods -n huggingface
oc logs build/custom-mnist-image-1 -n huggingface -f
oc wait --for=condition=complete build.build.openshift.io/custom-mnist-image-1 -n huggingface --timeout=600s
oc delete bc -n huggingface --selector build=custom-mnist-image
oc delete is -n huggingface --selector build=custom-mnist-image
```

Use the image after it is built image-registry.openshift-image-registry.svc:5000/huggingface/custom-mnist-image:latest
