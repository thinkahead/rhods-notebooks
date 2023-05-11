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
```


```
    opendatahub.io/notebook-image-name:
      "Alexei Custom Notebook"
    opendatahub.io/notebook-image-desc: "Alexei Custom Jupyter notebook image"
  labels:
    opendatahub.io/notebook-image: 'true'
```

```
        - name: JUPYTER_IMAGE
          value: image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/tensorflow-notebook:latest
        image: image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/tensorflow-notebook:latest
```

```
oc adm policy add-scc-to-user anyuid system:serviceaccount:generative-ai:stable-diffusion -n generative-ai
```
