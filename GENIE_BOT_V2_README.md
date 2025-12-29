# Genie Bot V2 - Usando databricks-langchain GenieAgent

Bot minimalista que usa `GenieAgent` do `databricks-langchain` para integração oficial com Genie Space, seguindo o padrão da documentação oficial do Databricks.

## Diferenças entre V1 e V2

### V1 (genie_bot.ipynb)
- ❌ Chamada manual à API do Genie via `requests`
- ❌ Autenticação manual via `WorkspaceClient.config.token`
- ❌ Parsing manual de responses
- ❌ Não usa ferramentas oficiais do Databricks
- ⚠️ Propenso a erros de autenticação e parsing

### V2 (genie_bot_v2.py) ✅ **RECOMENDADO**
- ✅ Usa `databricks_langchain.GenieAgent` (ferramenta oficial)
- ✅ Autenticação automática gerenciada pelo Databricks
- ✅ Parsing de responses automático
- ✅ Suporte a LangChain e LangGraph (extensível)
- ✅ Segue padrão da documentação oficial
- ✅ MLflow autolog integrado

## Arquivos

1. **[genie_bot_v2.py](genie_bot_v2.py)** - Notebook completo de deployment
2. **[langgraph-multiagent-genie.py](langgraph-multiagent-genie.py)** - Referência oficial Databricks

## Pré-requisitos

- ✅ Genie Space criado e funcional
- ✅ SQL Warehouse configurado
- ✅ Tabelas configuradas no Genie Space
- ✅ Acesso ao endpoint LLM `databricks-llama-4-maverick`

## Configuração

### Step 1: Obter IDs necessários

1. **Genie Space ID**:
   - Acesse seu Genie Space
   - URL: `https://.../genie/rooms/{SPACE_ID}`
   - Copie o `{SPACE_ID}`

2. **SQL Warehouse ID**:
   - Vá para SQL Warehouses no Databricks
   - Selecione o warehouse usado pelo Genie
   - Copie o ID da URL ou configurações

3. **Tabelas**:
   - Identifique as tabelas principais usadas pelo Genie
   - Formato: `catalog.schema.table`

### Step 2: Editar genie_bot_v2.py

No arquivo `genie_bot_agent.py` (gerado pelo `%%writefile`), edite:

```python
# Linha ~26
GENIE_SPACE_ID = "seu-space-id-aqui"

# Na seção de log do modelo (célula de resources):
WAREHOUSE_ID = "seu-warehouse-id"
CATALOG = "hs_franquia"
SCHEMA = "gold_connect_bot"
TABLE = "sua_tabela_principal"
```

### Step 3: Executar o Notebook

1. Upload `genie_bot_v2.py` para Databricks workspace
2. Execute as células em ordem:
   - **Cell 1**: Instala dependências
   - **Cell 2**: Define agent (cria `genie_bot_agent.py`)
   - **Cell 3-4**: Testa localmente
   - **Cell 5**: Loga model no MLflow
   - **Cell 6**: Validação pre-deployment
   - **Cell 7**: Registra no Unity Catalog
   - **Cell 8**: Deploy

## Como Funciona

### GenieAgent
```python
genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="query_genie",
    description="Agent that queries CRM data..."
)
```

O `GenieAgent`:
- É um LangChain Runnable
- Gerencia autenticação automaticamente
- Faz parsing de responses
- Retorna resultados estruturados
- Suporta message processors customizados

### Autenticação Automática

Ao declarar recursos no `mlflow.pyfunc.log_model()`:
```python
resources = [
    DatabricksGenieSpace(genie_space_id=GENIE_SPACE_ID),
    DatabricksSQLWarehouse(warehouse_id=WAREHOUSE_ID),
    DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.{TABLE}"),
]
```

O Databricks:
- Provisiona credenciais de curta duração automaticamente
- Rotaciona credenciais automaticamente
- Gerencia acesso seguro aos recursos
- Não requer tokens manuais

## Arquitetura

```
User Query → MLflow Endpoint → GenieBot.predict()
  ↓
GenieAgent.invoke() → Genie Space API (autenticação automática)
  ↓
Genie retorna SQL + resultado
  ↓
GenieAgent faz parsing automático
  ↓
Response para usuário
```

## Vantagens da V2

1. **Simplicidade**: Menos código, mais robusto
2. **Manutenção**: Usa ferramentas oficiais que recebem updates
3. **Segurança**: Autenticação gerenciada pelo Databricks
4. **Extensibilidade**: Pode adicionar mais agents com LangGraph
5. **Observabilidade**: MLflow autolog captura traces automaticamente
6. **Documentação**: Padrão oficial com suporte completo

## Extensibilidade Futura

Com V2, é fácil adicionar:

### Multi-Agent System
```python
from langgraph_supervisor import create_supervisor

agents = [
    genie_agent,
    vector_search_agent,
    custom_tool_agent,
]

supervisor = create_supervisor(agents=agents, model=llm)
```

### Message Processor Customizado
```python
def custom_processor(messages):
    # Filtrar mensagens antes de enviar ao Genie
    return filtered_messages

genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    message_processor=custom_processor
)
```

### Integração com Vector Search
```python
from databricks_langchain import VectorSearchRetrieverTool

vector_tool = VectorSearchRetrieverTool(
    index_name="catalog.schema.index"
)
```

## Troubleshooting

### Erro: Genie Space not found
- Verifique se o `GENIE_SPACE_ID` está correto
- Confirme que você tem acesso ao Genie Space

### Erro: SQL Warehouse permission denied
- Verifique o `WAREHOUSE_ID`
- Confirme permissões no SQL Warehouse
- Certifique-se que o warehouse está rodando

### Erro: Table not found
- Verifique formato: `catalog.schema.table`
- Confirme que a tabela existe
- Verifique permissões na tabela

## Migração de V1 para V2

Se você já tem V1 rodando:

1. Não precisa deletar V1 (pode manter como fallback)
2. Deploy V2 com nome diferente: `genie_bot_v2`
3. Teste V2 em paralelo
4. Migre tráfego gradualmente
5. Deprecie V1 após validação

## Referências

- [Notebook oficial Databricks](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-multiagent-genie.html)
- [databricks-langchain GenieAgent](https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_langchain.html#databricks_langchain.GenieAgent)
- [Databricks Agent Framework](https://docs.databricks.com/generative-ai/agent-framework/)
- [MLflow ResponsesAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent)
