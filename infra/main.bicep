// The whole PoC in one resource group: Container Apps environment with a serverless A100
// profile, registry, identity, logs, and Foundry with the LLM deployment. Entra ID only.
targetScope = 'resourceGroup'

param location string = resourceGroup().location

@description('Object id of the user who runs the LLM track (gets Cognitive Services OpenAI User)')
param userObjectId string

@description('Serverless GPU workload profiles of the Container Apps environment')
param gpuProfiles array = [{ name: 'gpu-a100', type: 'Consumption-GPU-NC24-A100' }]

@description('LLM deployments, Data Zone Standard; capacity is in thousands of tokens per minute')
param llmDeployments array = [
  { name: 'gpt-5.6-luna', version: '2026-07-09', capacity: 300 }
  { name: 'gpt-6-luna', version: '2026-09-22', capacity: 300 }
]

var uniq = uniqueString(resourceGroup().id)
var acrPullRole = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
var openAIUserRole = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'

resource law 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-logpoc'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    workspaceCapping: { dailyQuotaGb: 1 }
  }
}

resource id 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-logpoc'
  location: location
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'crlogpoc${uniq}'
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: acr
  name: guid(acr.id, id.id, acrPullRole)
  properties: {
    principalId: id.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRole)
  }
}

resource cae 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-logpoc'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: law.properties.customerId
        sharedKey: law.listKeys().primarySharedKey
      }
    }
    workloadProfiles: concat(
      [{ name: 'Consumption', workloadProfileType: 'Consumption' }],
      map(gpuProfiles, p => { name: p.name, workloadProfileType: p.type })
    )
  }
}

resource aif 'Microsoft.CognitiveServices/accounts@2025-06-01' = {
  name: 'aif-logpoc-${uniq}'
  location: location
  kind: 'AIServices'
  sku: { name: 'S0' }
  identity: { type: 'SystemAssigned' }
  properties: {
    customSubDomainName: 'aif-logpoc-${uniq}'
    allowProjectManagement: true
    disableLocalAuth: true
    publicNetworkAccess: 'Enabled'
  }
}

resource project 'Microsoft.CognitiveServices/accounts/projects@2025-06-01' = {
  parent: aif
  name: 'logpoc'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {}
}

// one at a time: parallel deployment writes on one account conflict
@batchSize(1)
resource llm 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = [
  for d in llmDeployments: {
    parent: aif
    name: d.name
    sku: { name: 'DataZoneStandard', capacity: d.capacity }
    properties: {
      model: { format: 'OpenAI', name: d.name, version: d.version }
    }
    dependsOn: [project]
  }
]

resource userLlm 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: aif
  name: guid(aif.id, userObjectId, openAIUserRole)
  properties: {
    principalId: userObjectId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', openAIUserRole)
  }
}

output acrName string = acr.name
output environmentName string = cae.name
output foundryEndpoint string = 'https://${aif.properties.customSubDomainName}.openai.azure.com/'
output llmDeployments array = [for (d, i) in llmDeployments: llm[i].name]
