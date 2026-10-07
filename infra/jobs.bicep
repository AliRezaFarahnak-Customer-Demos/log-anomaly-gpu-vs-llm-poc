// The GPU job: download the base model from Hugging Face, LoRA fine-tune on normal traces,
// score the labelled test set, and print the run record and metrics as one RESULT log line.
// Uses the replica's local disk: new storage accounts in this tenant get shared key and public
// network access switched off, so an Azure Files mount is not possible.
targetScope = 'resourceGroup'

param location string = resourceGroup().location

@description('Image tag in the registry')
param imageTag string

param gpuProfile string = 'gpu-a100'
param config string = 'configs/llama2_7b.yaml'
param data string = 'data/synthetic/prod_like'

var uniq = uniqueString(resourceGroup().id)

resource cae 'Microsoft.App/managedEnvironments@2024-03-01' existing = {
  name: 'cae-logpoc'
}

resource id 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: 'id-logpoc'
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: 'crlogpoc${uniq}'
}

var pipeline = '''
set -e
nvidia-smi -L || true
python - <<'EOF'
import os, yaml
from pathlib import Path
from logpoc.poc1_gpu.download_model import download
b = yaml.safe_load(open(os.environ["CONFIG"]))["base_model"]
download(b["repo_id"], Path(os.environ["ML_ROOT"]) / b["local_dir"])
EOF
python -m logpoc train --data "$DATA"
python -m logpoc evaluate --run-id "$RUN_ID" --data "$DATA" --split test --device cuda
python - <<'EOF'
import json, os
r = os.path.join(os.environ["ML_ROOT"], "runs", os.environ["RUN_ID"])
files = ("meta.json", "train_metrics.json", "eval/metrics.json")
print("RESULT " + json.dumps({f: json.load(open(os.path.join(r, f))) for f in files}))
EOF
'''

resource run 'Microsoft.App/jobs@2024-03-01' = {
  name: 'job-run'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${id.id}': {} }
  }
  properties: {
    environmentId: cae.id
    workloadProfileName: gpuProfile
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 21600
      replicaRetryLimit: 0
      manualTriggerConfig: {
        parallelism: 1
        replicaCompletionCount: 1
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: id.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'logpoc'
          image: '${acr.properties.loginServer}/logpoc:${imageTag}'
          command: ['/bin/sh', '-c']
          args: [replace(pipeline, '\r', '')]
          resources: { cpu: 24, memory: '220Gi' }
          env: [
            { name: 'ML_ROOT', value: '/tmp/ml' }
            { name: 'HF_HOME', value: '/tmp/ml/hf-cache' }
            { name: 'CONFIG', value: config }
            { name: 'DATA', value: data }
            { name: 'RUN_ID', value: 'unset' }
          ]
        }
      ]
    }
  }
}
