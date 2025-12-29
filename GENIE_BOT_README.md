# Genie Bot MVP - Guia de Deployment

Bot minimalista que encaminha queries para Databricks Genie Space via MLflow endpoint.

## Arquivos Criados

1. **[genie_bot.py](genie_bot.py)** - ResponsesAgent com integração Genie
2. **[deploy_genie_bot.ipynb](deploy_genie_bot.ipynb)** - Notebook de deployment
3. **[create_uc_function_query_genie.sql](create_uc_function_query_genie.sql)** - UC Function para Genie API

## Pré-requisitos

- ✅ Genie Space criado e funcional
- ✅ Acesso ao Databricks workspace
- ✅ Permissões para criar UC Functions em `hs_franquia.gold_connect_bot`
- ✅ Acesso ao endpoint LLM `databricks-llama-4-maverick`

## Passo a Passo

### Step 1: Obter o Genie Space ID

1. Acesse seu Genie Space no Databricks
2. A URL terá o formato: `https://.../genie/rooms/{SPACE_ID}`
3. Copie o `SPACE_ID`

### Step 2: Criar UC Function

1. Abra o SQL Editor ou um notebook SQL no Databricks
2. Execute o conteúdo de [create_uc_function_query_genie.sql](create_uc_function_query_genie.sql)
3. Substitua `SEU_SPACE_ID_AQUI` pelo Space ID copiado no Step 1
4. Execute o teste no final do script para validar

**Validação esperada**:
```sql
SELECT hs_franquia.gold_connect_bot.query_genie(
  'Quantas verificações tivemos em 2024?',
  'seu-space-id'
);
```
Deve retornar uma resposta do Genie.

### Step 3: Configurar genie_bot.py

1. Edite [genie_bot.py](genie_bot.py)
2. Linha 22: Substitua `SEU_SPACE_ID_AQUI` pelo seu Space ID
   ```python
   GENIE_SPACE_ID = "seu-space-id-aqui"
   ```

### Step 4: Deploy via Notebook

1. Upload [genie_bot.py](genie_bot.py) e [deploy_genie_bot.ipynb](deploy_genie_bot.ipynb) para o Databricks workspace
2. Abra [deploy_genie_bot.ipynb](deploy_genie_bot.ipynb)
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

### Erro: UC Function não encontrada

**Problema**: `Function hs_franquia.gold_connect_bot.query_genie not found`

**Solução**:
- Verifique se a UC Function foi criada corretamente
- Execute: `SHOW FUNCTIONS IN hs_franquia.gold_connect_bot LIKE 'query_genie'`
- Se não aparecer, re-execute [create_uc_function_query_genie.sql](create_uc_function_query_genie.sql)

### Erro: Genie API timeout

**Problema**: Query demora muito e timeout após 30s

**Solução**:
- Aumente o timeout na UC Function (linha com `timeout=30`)
- Simplifique a query para o Genie
- Verifique se o Genie Space está respondendo

### Erro: Permission denied

**Problema**: `User does not have permission to access...`

**Solução**:
- Verifique permissões no schema `hs_franquia.gold_connect_bot`
- Verifique acesso ao Genie Space
- Verifique acesso ao endpoint LLM

### Agent não está chamando a tool

**Problema**: LLM responde diretamente sem usar query_genie

**Solução**:
- Verifique se a UC Function está listada em `UC_TOOL_NAMES`
- Ajuste o SYSTEM_PROMPT para ser mais explícito
- Teste com queries mais claras sobre dados

## Próximos Passos (Features Futuras)

Após validar o MVP funcionando:

1. **Access Control**: Adicionar UC Function `check_user_access`
2. **Conversation History**: UC Functions `save_conversation` e `get_conversation_context`
3. **Large Result Uploads**: UC Function `upload_large_result` para datasets grandes
4. **Vector Search**: Adicionar busca em documentação via Vector Search
5. **Testes**: Criar suite de testes automatizados
6. **Monitoramento**: Adicionar métricas customizadas e alertas

## Arquitetura

```
User Query → MLflow Endpoint → ResponsesAgent.predict()
  ↓
LLM decide chamar tool query_genie
  ↓
UC Function query_genie → Genie Space API
  ↓
Genie retorna SQL + resultado
  ↓
LLM formata resposta final
  ↓
Response para usuário
```

## Referências

- [driver.ipynb](driver.ipynb) - Template original do ResponsesAgent
- [Databricks Agent Framework Docs](https://docs.databricks.com/generative-ai/agent-framework/)
- [MLflow ResponsesAgent Docs](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent)
