# Genie Bot MVP - Guia de Deployment

Bot minimalista que encaminha queries para Databricks Genie Space via MLflow endpoint usando chamada direta à API (sem UC Functions).

## Arquivos Criados

1. **[genie_bot.ipynb](genie_bot.ipynb)** - Notebook de desenvolvimento e testes
2. **[deploy_genie_bot.ipynb](deploy_genie_bot.ipynb)** - Notebook de deployment

## Pré-requisitos

- ✅ Genie Space criado e funcional
- ✅ Acesso ao Databricks workspace (WorkspaceClient cuida da autenticação automaticamente)
- ✅ Acesso ao endpoint LLM `databricks-llama-4-maverick`

## Passo a Passo

### Step 1: Obter o Genie Space ID

1. Acesse seu Genie Space no Databricks
2. A URL terá o formato: `https://.../genie/rooms/{SPACE_ID}`
3. Copie o `SPACE_ID`

### Step 2: Configurar genie_bot.ipynb

1. Upload [genie_bot.ipynb](genie_bot.ipynb) para o Databricks workspace
2. Abra o notebook
3. Na célula de configurações, substitua `SEU_SPACE_ID_AQUI` pelo seu Space ID:
   ```python
   GENIE_SPACE_ID = "seu-space-id-aqui"
   ```

### Step 3: Testar localmente no notebook

1. Execute as células em ordem:
   - **Cell 0**: Instala dependências
   - **Cells 1-6**: Define imports, configurações, tool, specs, classe GenieBot, e inicializa agent
   - **Cells 7-13**: Testes diversos (simple query, streaming, multiple queries, etc.)

2. Valide que as respostas do Genie estão corretas

### Step 4: Deploy via deploy_genie_bot.ipynb

1. Upload [deploy_genie_bot.ipynb](deploy_genie_bot.ipynb) para o Databricks workspace
2. Abra o notebook
3. Execute as células sequencialmente:
   - **Cell 1**: Instala dependências
   - **Cell 2**: Testa localmente
   - **Cell 3**: Loga model no MLflow
   - **Cell 4**: Registra no Unity Catalog
   - **Cell 5**: Deploy para serving endpoint

### Step 5: Validar Deployment

1. Acesse **AI Playground** no Databricks
2. Selecione o modelo `hs_franquia.gold_connect_bot.genie_bot`
3. Teste com queries:
   - "Quantas verificações tivemos em 2024?"
   - "Me mostre os riscos com mais desvios"
   - "Quais são os top 10 usuários com mais ações abertas?"

## Testando via API

Após deployment, você pode chamar o endpoint via REST API:

```python
import requests
import json

workspace_url = "https://sua-workspace.azuredatabricks.net"
token = "seu-token-aqui"

url = f"{workspace_url}/serving-endpoints/genie_bot/invocations"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

payload = {
    "input": [{
        "role": "user",
        "content": "Quantas verificações em 2024?"
    }]
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

## Troubleshooting

### Erro: Genie API timeout

**Problema**: Query demora muito e timeout após 60s

**Solução**:
- Aumente o timeout na função `query_genie_tool` (linha com `timeout=60`)
- Simplifique a query para o Genie
- Verifique se o Genie Space está respondendo

### Erro: Permission denied

**Problema**: `User does not have permission to access...`

**Solução**:
- Verifique acesso ao Genie Space
- Verifique acesso ao endpoint LLM `databricks-llama-4-maverick`
- Verifique se o WorkspaceClient está autenticado corretamente

### Agent não está chamando a tool

**Problema**: LLM responde diretamente sem usar query_genie

**Solução**:
- Verifique se `TOOL_INFOS` foi criado corretamente
- Ajuste o SYSTEM_PROMPT para ser mais explícito
- Teste com queries mais claras sobre dados
- Verifique se a tool spec está sendo passada corretamente para o LLM

### Erro: NameError com ToolInfo

**Problema**: `NameError: name 'ToolInfo' is not defined`

**Solução**:
- Verifique se todas as células foram executadas em ordem
- Re-execute a célula que importa `from pydantic import BaseModel`
- Re-execute a célula que define a classe `ToolInfo`

## Próximos Passos (Features Futuras)

Após validar o MVP funcionando:

1. **Access Control**: Adicionar controle de acesso por usuário/grupo
2. **Conversation History**: Persistir histórico de conversas
3. **Large Result Uploads**: Gerenciar datasets grandes via upload
4. **Vector Search**: Adicionar busca em documentação via Vector Search
5. **Testes**: Criar suite de testes automatizados
6. **Monitoramento**: Adicionar métricas customizadas e alertas

## Arquitetura

```
User Query → MLflow Endpoint → ResponsesAgent.predict()
  ↓
LLM decide chamar tool query_genie
  ↓
query_genie_tool (Python function) → Genie Space API
  ↓  (autenticação via WorkspaceClient)
Genie retorna SQL + resultado
  ↓
LLM formata resposta final
  ↓
Response para usuário
```

## Diferenças em relação ao driver.ipynb

Este bot segue o mesmo padrão do [driver.ipynb](driver.ipynb), mas com simplificações:

1. **Sem UC Functions**: Chama a API do Genie diretamente via Python (não via UC Function)
2. **Autenticação automática**: WorkspaceClient cuida de toda autenticação
3. **Uma única tool**: Apenas `query_genie` que encaminha para Genie Space
4. **Menos dependências**: Não precisa de `unitycatalog-ai` para execução de UC Functions

## Referências

- [driver.ipynb](driver.ipynb) - Template original do ResponsesAgent
- [Databricks Agent Framework Docs](https://docs.databricks.com/generative-ai/agent-framework/)
- [MLflow ResponsesAgent Docs](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent)
