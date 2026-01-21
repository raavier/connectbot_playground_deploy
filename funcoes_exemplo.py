# Databricks notebook source
# MAGIC %md
# MAGIC # Documentação de Funções - Genie Bot v2
# MAGIC
# MAGIC Este notebook documenta todos os parâmetros das funções e classes utilizadas no `genie_bot_v2.py`.
# MAGIC Inclui exemplos de uso para cada parâmetro.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. GenieAgent (databricks_langchain.genie)
# MAGIC
# MAGIC O `GenieAgent` cria um runnable LangChain que interage com o Databricks Genie Space.
# MAGIC
# MAGIC ### Assinatura
# MAGIC ```python
# MAGIC GenieAgent(
# MAGIC     genie_space_id: str,                    # OBRIGATÓRIO
# MAGIC     genie_agent_name: str = 'Genie',        # Nome do agente
# MAGIC     description: str = '',                   # Descrição para uso em multi-agent
# MAGIC     include_context: bool = False,          # Incluir contexto na query
# MAGIC     message_processor: Callable | None = None,  # Função para processar mensagens
# MAGIC     client: WorkspaceClient | None = None,  # Cliente Databricks customizado
# MAGIC     return_pandas: bool = False             # Retornar DataFrames pandas
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros do GenieAgent

# COMMAND ----------

from databricks_langchain.genie import GenieAgent

# -----------------------------------------------------------
# genie_space_id (str) - OBRIGATÓRIO
# ID do Genie Space no Databricks
# -----------------------------------------------------------
GENIE_SPACE_ID = "01f0ebc08dc313b1ace704b3d459b2bf"

# Exemplo básico
genie_agent_basic = GenieAgent(
    genie_space_id=GENIE_SPACE_ID
)

# -----------------------------------------------------------
# genie_agent_name (str) - default: 'Genie'
# Nome do agente usado em sistemas multi-agent
# -----------------------------------------------------------
genie_agent_named = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="query_genie_gold"  # Nome customizado
)

# -----------------------------------------------------------
# description (str) - default: ''
# Descrição do agente para contexto em multi-agent systems
# -----------------------------------------------------------
genie_agent_described = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="query_genie_gold",
    description="Agente que consulta dados do CRM usando linguagem natural. Use para perguntas sobre verificações, riscos, desvios e métricas."
)

# -----------------------------------------------------------
# include_context (bool) - default: False
# Quando True, inclui contexto adicional na query
# -----------------------------------------------------------
genie_agent_with_context = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    include_context=True
)

# -----------------------------------------------------------
# return_pandas (bool) - default: False
# Quando True, retorna resultados como DataFrame pandas
# -----------------------------------------------------------
genie_agent_pandas = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    return_pandas=True  # Útil para processamento de dados
)

# -----------------------------------------------------------
# client (WorkspaceClient | None) - default: None
# Cliente Databricks customizado para autenticação
# -----------------------------------------------------------
from databricks.sdk import WorkspaceClient

# Usando cliente padrão (None usa autenticação automática)
genie_agent_default_client = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    client=None  # Usa autenticação automática do ambiente
)

# Usando cliente customizado
custom_client = WorkspaceClient(
    host="https://your-workspace.databricks.com",
    token="dapi..."
)
genie_agent_custom_client = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    client=custom_client
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### message_processor - Exemplos Detalhados
# MAGIC
# MAGIC O `message_processor` é uma função que processa mensagens antes de enviar ao Genie.
# MAGIC Recebe uma lista de mensagens e retorna uma string de query.

# COMMAND ----------

# -----------------------------------------------------------
# EXEMPLO 1: Usar apenas a última mensagem
# -----------------------------------------------------------
def last_message_only(messages):
    """Envia apenas a última mensagem do usuário para o Genie"""
    if not messages:
        return ""
    last_msg = messages[-1]
    if isinstance(last_msg, dict):
        return last_msg.get("content", "")
    return last_msg.content

genie_agent_last_msg = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="query_genie_gold",
    message_processor=last_message_only,
    return_pandas=True
)

# -----------------------------------------------------------
# EXEMPLO 2: Filtrar apenas mensagens do usuário
# -----------------------------------------------------------
def user_messages_only(messages):
    """Concatena apenas mensagens do role=user"""
    user_msgs = []
    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                user_msgs.append(msg.get("content", ""))
        else:
            if msg.role == "user":
                user_msgs.append(msg.content)
    return " | ".join(user_msgs)

genie_agent_user_only = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    message_processor=user_messages_only
)

# -----------------------------------------------------------
# EXEMPLO 3: Adicionar contexto fixo à query
# -----------------------------------------------------------
def add_context_processor(messages):
    """Adiciona contexto temporal às queries"""
    if not messages:
        return ""

    last_msg = messages[-1]
    content = last_msg.get("content", "") if isinstance(last_msg, dict) else last_msg.content

    # Adiciona escopo temporal
    return f"Considerando dados de 2024 e 2025: {content}"

genie_agent_with_context = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    message_processor=add_context_processor
)

# -----------------------------------------------------------
# EXEMPLO 4: Limpar e normalizar a query
# -----------------------------------------------------------
import re

def clean_query_processor(messages):
    """Remove caracteres especiais e normaliza"""
    if not messages:
        return ""

    last_msg = messages[-1]
    content = last_msg.get("content", "") if isinstance(last_msg, dict) else last_msg.content

    # Remove caracteres especiais problemáticos
    cleaned = re.sub(r'[^\w\s?áéíóúãõâêôçÁÉÍÓÚÃÕÂÊÔÇ]', '', content)
    return cleaned.strip()

genie_agent_cleaned = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    message_processor=clean_query_processor
)

# -----------------------------------------------------------
# EXEMPLO 5: Usar últimas N mensagens (factory pattern)
# -----------------------------------------------------------
def last_n_messages(n=3):
    """Factory que retorna processor para últimas N mensagens"""
    def processor(messages):
        if not messages:
            return ""

        recent = messages[-n:] if len(messages) >= n else messages
        parts = []
        for msg in recent:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = msg.role
                content = msg.content
            parts.append(f"{role}: {content}")

        return "\n".join(parts)
    return processor

genie_agent_last_3 = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    message_processor=last_n_messages(n=3)  # últimas 3 mensagens
)

# -----------------------------------------------------------
# EXEMPLO 6: Filtrar por palavras-chave
# -----------------------------------------------------------
def keyword_filter_processor(keywords):
    """Factory que filtra mensagens contendo palavras-chave"""
    def processor(messages):
        filtered = []
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else msg.content
            if any(kw.lower() in content.lower() for kw in keywords):
                filtered.append(content)
        return " ".join(filtered) if filtered else messages[-1].get("content", "") if messages else ""
    return processor

genie_agent_keyword = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    message_processor=keyword_filter_processor(["verificação", "risco", "ação"])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. ChatDatabricks (databricks_langchain)
# MAGIC
# MAGIC Cliente LangChain para chamar modelos LLM hospedados no Databricks Model Serving.
# MAGIC
# MAGIC ### Assinatura
# MAGIC ```python
# MAGIC ChatDatabricks(
# MAGIC     endpoint: str = None,           # Nome do endpoint
# MAGIC     model: str = None,              # Alias para endpoint
# MAGIC     temperature: float = 0.0,       # Temperatura de sampling
# MAGIC     max_tokens: int = None,         # Máximo de tokens na resposta
# MAGIC     timeout: float = None,          # Timeout em segundos
# MAGIC     max_retries: int = 2,           # Número de retentativas
# MAGIC     stop: List[str] = None,         # Sequências de parada
# MAGIC     extra_params: dict = None,      # Parâmetros extras
# MAGIC     n: int = 1,                     # Número de completions
# MAGIC     target_uri: str = "databricks"  # URI alvo
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros do ChatDatabricks

# COMMAND ----------

from databricks_langchain import ChatDatabricks

# -----------------------------------------------------------
# endpoint (str) - Nome do Serving Endpoint
# -----------------------------------------------------------
llm_basic = ChatDatabricks(
    endpoint="databricks-llama-4-maverick"
)

# -----------------------------------------------------------
# model (str) - Alias para endpoint (mais legível)
# -----------------------------------------------------------
llm_model = ChatDatabricks(
    model="databricks-claude-3-7-sonnet"
)

# -----------------------------------------------------------
# temperature (float) - default: 0.0
# Controla aleatoriedade. 0=determinístico, 1=criativo
# -----------------------------------------------------------
llm_creative = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    temperature=0.7  # Mais criativo
)

llm_deterministic = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    temperature=0.0  # Respostas consistentes
)

# -----------------------------------------------------------
# max_tokens (int) - Máximo de tokens na resposta
# -----------------------------------------------------------
llm_short = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    max_tokens=100  # Respostas curtas
)

llm_long = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    max_tokens=4096  # Respostas longas
)

# -----------------------------------------------------------
# timeout (float) - Timeout em segundos
# -----------------------------------------------------------
llm_timeout = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    timeout=30.0  # 30 segundos
)

# -----------------------------------------------------------
# max_retries (int) - Número de retentativas em caso de falha
# -----------------------------------------------------------
llm_retries = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    max_retries=5  # Mais resiliente
)

# -----------------------------------------------------------
# stop (List[str]) - Sequências que param a geração
# -----------------------------------------------------------
llm_stop = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    stop=["###", "FIM", "\n\n"]  # Para quando encontrar esses tokens
)

# -----------------------------------------------------------
# extra_params (dict) - Parâmetros extras para o endpoint
# -----------------------------------------------------------
llm_extra = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    extra_params={
        "top_p": 0.9,
        "frequency_penalty": 0.5
    }
)

# -----------------------------------------------------------
# n (int) - Número de completions a gerar
# -----------------------------------------------------------
llm_multiple = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    n=3  # Gera 3 respostas alternativas
)

# -----------------------------------------------------------
# Exemplo completo com todos os parâmetros
# -----------------------------------------------------------
llm_full = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    temperature=0.1,
    max_tokens=500,
    timeout=60.0,
    max_retries=3,
    stop=["###"],
    extra_params={"top_p": 0.95}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. ResponsesAgent (mlflow.pyfunc)
# MAGIC
# MAGIC Classe base para criar agentes compatíveis com o Databricks Agent Framework.
# MAGIC
# MAGIC ### Métodos a implementar
# MAGIC ```python
# MAGIC class MyAgent(ResponsesAgent):
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         """Processa request e retorna response (não-streaming)"""
# MAGIC         pass
# MAGIC
# MAGIC     def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent]:
# MAGIC         """Processa request e retorna stream de eventos"""
# MAGIC         pass
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros do ResponsesAgentRequest

# COMMAND ----------

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

# -----------------------------------------------------------
# ResponsesAgentRequest - Parâmetros principais
# -----------------------------------------------------------

# input (List[Message]) - OBRIGATÓRIO
# Lista de mensagens da conversa
input_example = {
    "input": [
        {"role": "system", "content": "Você é um assistente útil."},
        {"role": "user", "content": "Quantas verificações em 2024?"}
    ]
}

# custom_inputs (dict) - Parâmetros customizados
input_with_custom = {
    "input": [{"role": "user", "content": "Pergunta"}],
    "custom_inputs": {
        "user_id": "12345",
        "department": "vendas"
    }
}

# context (ChatContext) - Contexto da conversa
input_with_context = {
    "input": [{"role": "user", "content": "Pergunta"}],
    "context": {
        "conversation_id": "conv-123",
        "user_id": "user-456"
    }
}

# temperature (float) - Temperatura para o modelo
input_with_temp = {
    "input": [{"role": "user", "content": "Pergunta"}],
    "temperature": 0.7
}

# max_output_tokens (int) - Limite de tokens na resposta
input_with_max_tokens = {
    "input": [{"role": "user", "content": "Pergunta"}],
    "max_output_tokens": 500
}

# tools (List[Tool]) - Ferramentas disponíveis para o agente
input_with_tools = {
    "input": [{"role": "user", "content": "Pergunta"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Busca no banco de dados",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        }
    ]
}

# metadata (dict) - Metadados para tracking
input_with_metadata = {
    "input": [{"role": "user", "content": "Pergunta"}],
    "metadata": {
        "source": "web_app",
        "version": "1.0"
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exemplo completo de ResponsesAgent

# COMMAND ----------

from typing import Generator
from uuid import uuid4

class ExampleAgent(ResponsesAgent):
    """Exemplo de implementação de ResponsesAgent"""

    def __init__(self, llm, tools=None):
        self.llm = llm
        self.tools = tools or []

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Processa request de forma não-streaming.
        Coleta todos os eventos do stream e retorna resposta final.
        """
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs,
            custom_outputs=request.custom_inputs  # Passa inputs customizados
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Processa request com streaming.
        Emite eventos conforme processa.
        """
        # Extrair mensagens do request
        messages = request.input

        # Processar com LLM
        response = self.llm.invoke(messages)

        # Emitir evento de conclusão
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(
                text=response.content,
                id=str(uuid4())
            )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. mlflow.pyfunc.log_model
# MAGIC
# MAGIC Registra um modelo no MLflow com configurações para deploy.
# MAGIC
# MAGIC ### Assinatura
# MAGIC ```python
# MAGIC mlflow.pyfunc.log_model(
# MAGIC     name: str,                      # Nome do artefato
# MAGIC     python_model: str,              # Caminho do arquivo Python
# MAGIC     input_example: dict = None,     # Exemplo de input
# MAGIC     resources: List[Resource] = None,  # Recursos Databricks
# MAGIC     pip_requirements: List[str] = None,  # Dependências pip
# MAGIC     extra_pip_requirements: List[str] = None,  # Deps extras
# MAGIC     conda_env: dict = None,         # Ambiente conda
# MAGIC     code_paths: List[str] = None,   # Arquivos de código extras
# MAGIC     registered_model_name: str = None,  # Nome para registro
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros do log_model

# COMMAND ----------

import mlflow
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksFunction,
    DatabricksVectorSearchIndex,
)
from pkg_resources import get_distribution

# -----------------------------------------------------------
# name (str) - Nome do artefato do modelo
# -----------------------------------------------------------
# "genie_gold_crm_bot"

# -----------------------------------------------------------
# python_model (str) - Caminho do arquivo Python com o agente
# -----------------------------------------------------------
# "genie_bot_agent.py"

# -----------------------------------------------------------
# input_example (dict) - Exemplo de input para validação
# -----------------------------------------------------------
input_example = {
    "input": [
        {"role": "user", "content": "Quantas verificações em 2024?"}
    ]
}

# -----------------------------------------------------------
# resources (List[Resource]) - Recursos Databricks necessários
# -----------------------------------------------------------
resources = [
    # Endpoint de LLM
    DatabricksServingEndpoint(endpoint_name="databricks-llama-4-maverick"),

    # Genie Space
    DatabricksGenieSpace(genie_space_id="01f0ebc08dc313b1ace704b3d459b2bf"),

    # SQL Warehouse
    DatabricksSQLWarehouse(warehouse_id="4d81765d86b13aa1"),

    # Tabelas usadas
    DatabricksTable(table_name="hs_franquia.gold_crm.f_verificacao"),
    DatabricksTable(table_name="hs_franquia.gold_crm.f_acao"),

    # Funções UC usadas
    DatabricksFunction(function_name="hs_franquia.gold_connect_bot.resolve_hierarchy_from_text"),

    # Vector Search Index (se usar RAG)
    # DatabricksVectorSearchIndex(index_name="hs_franquia.gold_crm.docs_index"),
]

# -----------------------------------------------------------
# pip_requirements (List[str]) - Dependências pip específicas
# -----------------------------------------------------------
pip_requirements = [
    f"databricks-connect=={get_distribution('databricks-connect').version}",
    f"mlflow=={get_distribution('mlflow').version}",
    f"databricks-langchain=={get_distribution('databricks-langchain').version}",
]

# -----------------------------------------------------------
# extra_pip_requirements (List[str]) - Dependências extras
# -----------------------------------------------------------
extra_pip_requirements = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
]

# -----------------------------------------------------------
# Exemplo completo de log_model
# -----------------------------------------------------------
"""
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="genie_gold_crm_bot",
        python_model="genie_bot_agent.py",
        input_example=input_example,
        resources=resources,
        pip_requirements=pip_requirements,
    )
print(f"Model URI: {logged_agent_info.model_uri}")
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. MLflow Resources - Classes de Recursos
# MAGIC
# MAGIC Classes para declarar recursos Databricks necessários para o modelo.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Todos os tipos de Resources disponíveis

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksGenieSpace,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksFunction,
    DatabricksVectorSearchIndex,
    DatabricksUCConnection,
)

# -----------------------------------------------------------
# DatabricksServingEndpoint
# Endpoint de Model Serving (LLM, embeddings, etc)
# -----------------------------------------------------------
endpoint_resource = DatabricksServingEndpoint(
    endpoint_name="databricks-llama-4-maverick"
)

# Múltiplos endpoints
endpoints = [
    DatabricksServingEndpoint(endpoint_name="databricks-llama-4-maverick"),
    DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),  # embeddings
]

# -----------------------------------------------------------
# DatabricksGenieSpace
# Genie Space para queries em linguagem natural
# -----------------------------------------------------------
genie_resource = DatabricksGenieSpace(
    genie_space_id="01f0ebc08dc313b1ace704b3d459b2bf"
)

# -----------------------------------------------------------
# DatabricksSQLWarehouse
# SQL Warehouse para execução de queries
# -----------------------------------------------------------
warehouse_resource = DatabricksSQLWarehouse(
    warehouse_id="4d81765d86b13aa1"
)

# -----------------------------------------------------------
# DatabricksTable
# Tabela do Unity Catalog
# -----------------------------------------------------------
table_resource = DatabricksTable(
    table_name="hs_franquia.gold_crm.f_verificacao"
)

# Com on_behalf_of_user (para autenticação delegada)
table_user_auth = DatabricksTable(
    table_name="hs_franquia.gold_crm.f_verificacao",
    # on_behalf_of_user=True  # Usa credenciais do usuário
)

# Múltiplas tabelas
CATALOG = "hs_franquia"
SCHEMA = "gold_crm"
TABLES = ["f_verificacao", "f_acao", "f_pergunta", "f_pessoa"]

table_resources = [
    DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.{table}")
    for table in TABLES
]

# -----------------------------------------------------------
# DatabricksFunction
# Função do Unity Catalog
# -----------------------------------------------------------
function_resource = DatabricksFunction(
    function_name="hs_franquia.gold_connect_bot.resolve_hierarchy_from_text"
)

# Múltiplas funções
FUNCTIONS = [
    "hs_franquia.gold_connect_bot.get_subordinate_locations",
    "hs_franquia.gold_connect_bot.get_subordinate_org_units",
    "hs_franquia.gold_connect_bot.resolve_hierarchy_from_text",
    "hs_franquia.gold_connect_bot.resolve_location_from_text",
]

function_resources = [
    DatabricksFunction(function_name=func)
    for func in FUNCTIONS
]

# -----------------------------------------------------------
# DatabricksVectorSearchIndex
# Índice de Vector Search para RAG
# -----------------------------------------------------------
vector_resource = DatabricksVectorSearchIndex(
    index_name="hs_franquia.gold_crm.docs_index"
)

# -----------------------------------------------------------
# DatabricksUCConnection
# Conexão do Unity Catalog (para external data)
# -----------------------------------------------------------
connection_resource = DatabricksUCConnection(
    connection_name="my_external_connection"
)

# -----------------------------------------------------------
# Exemplo: Lista completa de resources para um agente
# -----------------------------------------------------------
all_resources = [
    # LLM Endpoint
    DatabricksServingEndpoint(endpoint_name="databricks-llama-4-maverick"),

    # Genie Space
    DatabricksGenieSpace(genie_space_id="01f0ebc08dc313b1ace704b3d459b2bf"),

    # SQL Warehouse
    DatabricksSQLWarehouse(warehouse_id="4d81765d86b13aa1"),

    # Tabelas
    *[DatabricksTable(table_name=f"hs_franquia.gold_crm.{t}")
      for t in ["f_verificacao", "f_acao", "f_pergunta"]],

    # Funções
    *[DatabricksFunction(function_name=f)
      for f in FUNCTIONS],
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. databricks.agents.deploy
# MAGIC
# MAGIC Faz deploy do agente como endpoint no Databricks Model Serving.
# MAGIC
# MAGIC ### Assinatura
# MAGIC ```python
# MAGIC databricks.agents.deploy(
# MAGIC     model_name: str,                    # Nome do modelo no UC
# MAGIC     model_version: int,                 # Versão do modelo
# MAGIC     scale_to_zero: bool = False,        # Escalar para zero
# MAGIC     environment_vars: Dict[str, str] = None,  # Vars de ambiente
# MAGIC     instance_profile_arn: str = None,   # IAM role (AWS)
# MAGIC     tags: Dict[str, str] = None,        # Tags do endpoint
# MAGIC     workload_size: str = 'Small',       # Tamanho do workload
# MAGIC     endpoint_name: str = None,          # Nome customizado
# MAGIC     budget_policy_id: str = None,       # Policy de budget
# MAGIC     description: str = None,            # Descrição do endpoint
# MAGIC     deploy_feedback_model: bool = True, # Deploy modelo de feedback
# MAGIC ) → Deployment
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros do agents.deploy

# COMMAND ----------

from databricks import agents

# -----------------------------------------------------------
# model_name (str) - OBRIGATÓRIO
# Nome completo do modelo no Unity Catalog
# -----------------------------------------------------------
UC_MODEL_NAME = "hs_franquia.gold_connect_bot.genie_gold_crm_bot"

# -----------------------------------------------------------
# model_version (int) - OBRIGATÓRIO
# Versão do modelo a fazer deploy
# -----------------------------------------------------------
MODEL_VERSION = 1

# -----------------------------------------------------------
# scale_to_zero (bool) - default: False
# Quando True, escala para 0 instâncias quando ocioso
# Economiza custos mas aumenta latência inicial
# -----------------------------------------------------------

# Deploy com scale to zero (economia de custos)
"""
agents.deploy(
    UC_MODEL_NAME,
    MODEL_VERSION,
    scale_to_zero=True  # Desliga quando não usado
)
"""

# Deploy sempre ativo (baixa latência)
"""
agents.deploy(
    UC_MODEL_NAME,
    MODEL_VERSION,
    scale_to_zero=False  # Sempre pronto
)
"""

# -----------------------------------------------------------
# tags (Dict[str, str]) - Tags para o endpoint
# Útil para organização e billing
# -----------------------------------------------------------
"""
agents.deploy(
    UC_MODEL_NAME,
    MODEL_VERSION,
    tags={
        "endpointSource": "genie_gold_crm_bot",
        "team": "data-science",
        "environment": "production",
        "cost_center": "crm-analytics"
    }
)
"""

# -----------------------------------------------------------
# workload_size (str) - default: 'Small'
# Tamanho do workload: 'Small', 'Medium', 'Large'
# -----------------------------------------------------------
"""
# Para baixo volume
agents.deploy(
    UC_MODEL_NAME,
    MODEL_VERSION,
    workload_size='Small'
)

# Para alto volume
agents.deploy(
    UC_MODEL_NAME,
    MODEL_VERSION,
    workload_size='Large'
)
"""

# -----------------------------------------------------------
# endpoint_name (str) - Nome customizado do endpoint
# Se None, usa nome baseado no modelo
# -----------------------------------------------------------
"""
agents.deploy(
    UC_MODEL_NAME,
    MODEL_VERSION,
    endpoint_name="genie-crm-bot-prod"
)
"""

# -----------------------------------------------------------
# environment_vars (Dict[str, str]) - Variáveis de ambiente
# -----------------------------------------------------------
"""
agents.deploy(
    UC_MODEL_NAME,
    MODEL_VERSION,
    environment_vars={
        "LOG_LEVEL": "INFO",
        "MAX_RETRIES": "3"
    }
)
"""

# -----------------------------------------------------------
# description (str) - Descrição do endpoint
# -----------------------------------------------------------
"""
agents.deploy(
    UC_MODEL_NAME,
    MODEL_VERSION,
    description="Bot de CRM usando Genie Space para queries em linguagem natural"
)
"""

# -----------------------------------------------------------
# deploy_feedback_model (bool) - default: True
# Deploy modelo de feedback junto (deprecated Nov 2025)
# -----------------------------------------------------------
"""
agents.deploy(
    UC_MODEL_NAME,
    MODEL_VERSION,
    deploy_feedback_model=False  # Não deploy modelo de feedback
)
"""

# -----------------------------------------------------------
# Exemplo completo de deploy
# -----------------------------------------------------------
"""
deployment_info = agents.deploy(
    model_name="hs_franquia.gold_connect_bot.genie_gold_crm_bot",
    model_version=1,
    scale_to_zero=True,
    workload_size='Small',
    endpoint_name="genie-crm-bot-v2",
    tags={
        "endpointSource": "genie_gold_crm_bot",
        "team": "data-science",
        "version": "2.0"
    },
    description="Bot de CRM v2 com Genie Space"
)

print(f"Endpoint: {deployment_info.endpoint_name}")
print(f"Status: {deployment_info.status}")
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. mlflow.register_model
# MAGIC
# MAGIC Registra um modelo logado no Unity Catalog.
# MAGIC
# MAGIC ### Assinatura
# MAGIC ```python
# MAGIC mlflow.register_model(
# MAGIC     model_uri: str,     # URI do modelo logado
# MAGIC     name: str,          # Nome no Unity Catalog
# MAGIC     tags: dict = None,  # Tags do modelo
# MAGIC ) → ModelVersion
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parâmetros do register_model

# COMMAND ----------

import mlflow

# Configurar registry para Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# -----------------------------------------------------------
# model_uri (str) - URI do modelo logado
# Formatos: "runs:/<run_id>/<artifact_path>" ou "models:/<name>/<version>"
# -----------------------------------------------------------

# Usando run ID
# model_uri = f"runs:/{logged_agent_info.run_id}/genie_gold_crm_bot"

# Usando modelo registrado
# model_uri = "models:/hs_franquia.gold_connect_bot.genie_gold_crm_bot/1"

# -----------------------------------------------------------
# name (str) - Nome completo no Unity Catalog
# Formato: "<catalog>.<schema>.<model_name>"
# -----------------------------------------------------------
UC_MODEL_NAME = "hs_franquia.gold_connect_bot.genie_gold_crm_bot"

# -----------------------------------------------------------
# Exemplo de registro
# -----------------------------------------------------------
"""
# Primeiro, logar o modelo
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="genie_gold_crm_bot",
        python_model="genie_bot_agent.py",
        input_example=input_example,
        resources=resources,
    )

# Depois, registrar no Unity Catalog
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri,
    name=UC_MODEL_NAME
)

print(f"Registered: {UC_MODEL_NAME} version {uc_registered_model_info.version}")
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Funções Auxiliares do mlflow.types.responses
# MAGIC
# MAGIC Funções helper para trabalhar com ResponsesAgent.

# COMMAND ----------

from mlflow.types.responses import (
    to_chat_completions_input,
    output_to_responses_items_stream,
)

# -----------------------------------------------------------
# to_chat_completions_input
# Converte input do ResponsesAgent para formato ChatCompletion
# -----------------------------------------------------------

# Input no formato ResponsesAgent
responses_input = [
    {"role": "system", "content": "Você é um assistente."},
    {"role": "user", "content": "Olá!"}
]

# Converter para formato ChatCompletion
cc_msgs = to_chat_completions_input(responses_input)
# Resultado: lista de mensagens no formato OpenAI ChatCompletion

# -----------------------------------------------------------
# output_to_responses_items_stream
# Converte output para stream de ResponsesAgentStreamEvent
# -----------------------------------------------------------

# Usado para converter outputs de LLM para formato de streaming
# do ResponsesAgent

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Exemplo Completo: Genie Bot Configurável
# MAGIC
# MAGIC Exemplo que usa todos os parâmetros documentados.

# COMMAND ----------

# %%writefile genie_bot_configuravel.py
"""
Genie Bot Configurável - Exemplo com todos os parâmetros
"""
import json
from typing import Generator
from uuid import uuid4

import mlflow
from databricks_langchain import ChatDatabricks
from databricks_langchain.genie import GenieAgent
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

########################################
# Configuração
########################################

LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"
GENIE_SPACE_ID = "01f0ebc08dc313b1ace704b3d459b2bf"

SYSTEM_PROMPT = """Você é um assistente que responde perguntas sobre dados CRM usando o Genie Space.
Para qualquer pergunta sobre dados, use a ferramenta query_genie_gold para obter a resposta."""

########################################
# Message Processor Customizado
########################################

def smart_message_processor(messages):
    """
    Processor inteligente que:
    1. Pega apenas a última mensagem do usuário
    2. Remove caracteres especiais
    3. Adiciona contexto temporal
    """
    if not messages:
        return ""

    # Pegar última mensagem do usuário
    user_messages = [
        m for m in messages
        if (isinstance(m, dict) and m.get("role") == "user") or
           (hasattr(m, "role") and m.role == "user")
    ]

    if not user_messages:
        return ""

    last_msg = user_messages[-1]
    content = last_msg.get("content", "") if isinstance(last_msg, dict) else last_msg.content

    # Limpar caracteres especiais
    import re
    cleaned = re.sub(r'[^\w\s?áéíóúãõâêôçÁÉÍÓÚÃÕÂÊÔÇ,.]', '', content)

    return cleaned.strip()

########################################
# Criar Genie Agent com todos os parâmetros
########################################

genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="query_genie_gold",
    description="Agente que consulta dados do CRM usando linguagem natural",
    include_context=False,
    message_processor=smart_message_processor,  # Processor customizado
    client=None,  # Usa autenticação automática
    return_pandas=True
)

########################################
# Criar LLM com configurações otimizadas
########################################

llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT_NAME,
    temperature=0.1,  # Baixa para respostas consistentes
    max_tokens=1000,
    timeout=60.0,
    max_retries=3,
)

########################################
# ResponsesAgent Implementation
########################################

class GenieBotConfiguravel(ResponsesAgent):
    def __init__(self, genie_agent: GenieAgent, llm: ChatDatabricks):
        self.genie_agent = genie_agent
        self.llm = llm

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs,
            custom_outputs=request.custom_inputs
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])

        if SYSTEM_PROMPT:
            cc_msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        try:
            result = self.genie_agent.invoke({"messages": cc_msgs})

            if isinstance(result, dict) and "messages" in result:
                messages = result["messages"]
                if messages:
                    last_msg = messages[-1]
                    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=self.create_text_output_item(text=content, id=str(uuid4())),
                    )
            else:
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(text=str(result), id=str(uuid4())),
                )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text=error_msg, id=str(uuid4())),
            )

########################################
# Inicializar
########################################

mlflow.langchain.autolog()
AGENT = GenieBotConfiguravel(genie_agent=genie_agent, llm=llm)
mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Referências
# MAGIC
# MAGIC - [GenieAgent - databricks-langchain](https://github.com/databricks/databricks-ai-bridge)
# MAGIC - [ChatDatabricks - LangChain](https://python.langchain.com/docs/integrations/chat/databricks/)
# MAGIC - [ResponsesAgent - MLflow](https://mlflow.org/docs/latest/genai/serving/responses-agent/)
# MAGIC - [mlflow.pyfunc.log_model](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html)
# MAGIC - [agents.deploy - Databricks](https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent)
# MAGIC - [MLflow Resources](https://mlflow.org/docs/latest/_modules/mlflow/models/resources.html)
