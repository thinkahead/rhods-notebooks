Import the image and start the build
```
oc -n redhat-ods-applications import-image tensorflow-notebook:latest --from=docker.io/jupyter/tensorflow-notebook:latest --confirm
oc -n redhat-ods-applications new-build --name custom-generative-ai-image --code https://github.com/thinkahead/rhods-notebooks --context-dir generative_ai/custom-image
oc logs pod/custom-generative-ai-image-1-build -f -n redhat-ods-applications
```

Deleting the build
```
oc get bc -n redhat-ods-applications
oc delete bc custom-generative-ai-image -n redhat-ods-applications

Deleting the build
```
