#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#### Importando dados abertos da Receita Federal  ###

ds_rfb=pd.read_excel('gn-municipios - Tabela 12B.xlsx')


# In[ ]:


ds_rfb.info()


# In[ ]:


ds_rfb.info()


# In[ ]:


## Verificando se há dados faltantes
print(ds_rfb.isna().sum())


# In[ ]:


#### Tratamento dos dados abertos da Receita Federal ###
## limpando os nomes dos municípios
ds_rfb['Municipios'] = ds_rfb['Municipios'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
ds_rfb['Municipios'] = ds_rfb['Municipios'].apply(lambda x: x.upper())
ds_rfb.head()


# In[ ]:


## Agrupando os dados por modelo de formulário da declaração em cada município. 
## os municípios também passam a ser o índice da tabela.
ds_rfb2 = ds_rfb.groupby(['Municipios']).sum()
ds_rfb2.head()


# In[ ]:


## Verifica-se um total de 5.571 municípios nos dados da RFB
ds_rfb2.info()


# In[ ]:


## Calculando a média por declarantes e 
# multiplicando os valores por 1.000.000 (um milhão), para termos a ordem de grandeza real.

ds_rfb3 = ds_rfb2.apply(func = lambda x: (x*1000000)/ds_rfb2['Qtde Declarantes'])

## no passo anterior, o total de declarantes também foi dividido
## recuperando o valor original

ds_rfb3['Qtde Declarantes'] = ds_rfb2['Qtde Declarantes']


# In[ ]:


## Arrendondando os dados, com duas casas decimais
ds_rfb4 = ds_rfb3.round(decimals=2)
ds_rfb4.head()


# In[ ]:


#### Importando dados do IBGE ###
ds_pop = pd.read_excel('pop IBGE 2018.xlsx')
ds_pib = pd.read_excel('PIB munic 2018.xlsx')


# In[ ]:


## Verificando se há dados faltantes
print(ds_pop.isna().sum())


# In[ ]:


print(ds_pib.isna().sum())


# In[ ]:


## Observa-se que há dados faltantes na tabela do PIB, mas serão colunas que não serão utilizadas no modelo


# In[ ]:


ds_pop.info()


# In[ ]:


ds_pib.info()


# In[ ]:


## Destaca-se que os dados do IBGE possuem apenas 5.570 municípios - um a menos do que a tabela da RFB


# In[ ]:


#### Tratamento dos dados do IBGE ###
## Selecionando as colunas de interesse

ds_pop2 = ds_pop[['COD. UF','COD. MUNIC','NOME DO MUNICÍPIO','UF','POPULAÇÃO ESTIMADA']]
ds_pop2.head()


# In[ ]:


## Selecionando as colunas de interesse
# Renomeando coluna para 'PIB per capita'
# A coluna 'Atividade com maior valor adicionado bruto' será mantida para futura avaliação dos resultados

ds_pib2 = ds_pib[['Código da Unidade da Federação','Código do Município','Nome do Município','Sigla da Unidade da Federação','Produto Interno Bruto per capita, \na preços correntes\n(R$ 1,00)', 'Atividade com maior valor adicionado bruto']]
ds_pib2.rename(columns={'Produto Interno Bruto per capita, \na preços correntes\n(R$ 1,00)': 'PIB per capita'}, inplace=True)
ds_pib2.head()


# In[ ]:


## Limpando os nomes dos municípios
ds_pop2['NOME DO MUNICÍPIO'] = ds_pop2['NOME DO MUNICÍPIO'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
ds_pop2['NOME DO MUNICÍPIO'] = ds_pop2['NOME DO MUNICÍPIO'].apply(lambda x: x.upper())


# In[ ]:


## Limpando os nomes dos municípios
ds_pib2['Nome do Município'] = ds_pib2['Nome do Município'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
ds_pib2['Nome do Município'] = ds_pib2['Nome do Município'].apply(lambda x: x.upper())


# In[ ]:


# Criando uma coluna com os municípios no mesmo formato dos dados da RFB
ds_pop2['MUNICÍPIO_UF'] = ds_pop2['NOME DO MUNICÍPIO']+' - '+ds_pop2['UF']
ds_pib2['MUNICÍPIO_UF'] = ds_pib2['Nome do Município']+' - '+ds_pib2['Sigla da Unidade da Federação']


# In[ ]:


ds_pop2.head()


# In[ ]:


ds_pib2.head()


# In[ ]:


# Criando uma nova coluna de código do município na tabela PIB, para utilizar como junção com a tabela de População

ds_pib2['Cod_Municipio'] = ds_pib2['Código do Município'].astype(str).str[-5:].astype(int) 
ds_pib2.head(20)


# In[ ]:


## Junção dos dados do IBGE

ds_IBGE = ds_pib2.merge(ds_pop2, right_on=['COD. UF','COD. MUNIC'], left_on=['Código da Unidade da Federação','Cod_Municipio'])
ds_IBGE.info()


# In[ ]:


## Será utilizada a coluna de Municípios da tabela Pop, pois há menor diferença de grafias para a tabela RFB
ds_IBGE.rename(columns={'MUNICÍPIO_UF_y': 'MUNICÍPIO_UF'}, inplace=True)


# In[ ]:


ds_IBGE.head()


# In[ ]:


# Alterando o index para a coluna MUNICÍPIO_UF e selecionado as colunas de interesse
ds_IBGE.index = ds_IBGE['MUNICÍPIO_UF']
ds_IBGE2 = ds_IBGE[['PIB per capita','POPULAÇÃO ESTIMADA']]
ds_IBGE2.head()


# In[ ]:


## Verificando se há dados faltantes
print(ds_IBGE2.isna().sum())


# In[ ]:


### Junção dos dados IBGE e RFB ###
ds_completo = ds_rfb4.merge(ds_IBGE2, left_index=True, right_index=True)


# In[ ]:


## Verificando se há dados faltantes
print(ds_completo.isna().sum())


# In[ ]:


ds_completo.head()


# In[ ]:


ds_completo.info()


# In[ ]:


## Observa-se que a tabela final ficou com 5.541 municípios, pois há diferenças de grafia entre as tabelas RFB e IBGE


# In[ ]:


#### Análise exploratória dos dados completos ###
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


### Teste de Hipótese de Normalidade - Shapiro-Wilk ###

## O Teste de Shapiro não funciona bem para amostras maiores do que 5000 observações
## Assim, vamos selecionar linhas aleatoriamente 

ds_completo_rnd = ds_completo.sample(n=4800, random_state=1)

for valor in ds_completo_rnd.columns:
  nome = str(valor)
  x = ds_completo_rnd[nome]
  shapiro_stat, shapiro_p_valor = stats.shapiro(x)
  alpha = 0.05
  print("Para a coluna "+ nome + ", o valor da estatística de Shapiro-Wilk = " + str(shapiro_stat))
  print("Para a coluna "+ nome + ", o p_value Shapiro-Wilk = " + str(shapiro_p_valor))
  if shapiro_p_valor > alpha:
	  print("Com 95% de confiança, os dados de "+ nome + " são similiares a uma distribuição NORMAL (não foi possível rejeitar H0) \n")
  else:
	  print("Distribuição dos dados de " + nome + " NÃO é normal (H0 rejeitada) \n")
  



# In[ ]:


### Teste de Hipótese de Normalidade - Kolmogorov-Smirnov ###
from scipy.stats import kstest

for valor in ds_completo.columns:
  nome = str(valor)
  x = ds_completo[nome]
  ks_stat, ks_p_valor = kstest(x, 'norm')
  alpha = 0.05
  print("Para a coluna "+ nome + ", o valor da estatística de Kolmogorov-Smirnov = " + str(ks_stat))
  print("Para a coluna "+ nome + ", o p_value Kolmogorov-Smirnov = " + str(ks_p_valor))
  if ks_p_valor > alpha:
	  print("Com 95% de confiança, os dados de "+ nome + " são similiares a uma distribuição NORMAL (não foi possível rejeitar H0) \n")
  else:
	  print("Distribuição dos dados de " + nome + " NÃO é normal (H0 rejeitada) \n")


# In[ ]:


## Verifica-se que as variáveis não possuem distribuição normal


# In[ ]:


## Matriz de correlação de Pearson entre as variáveis
# Supõe que as variáveis possuem distribuição normal
fig, ax = plt.subplots(figsize=(20,16))
corr_pearson = sns.heatmap(ds_completo.corr(method='pearson'), annot=True, linewidths=.5)


# In[ ]:


## Matriz de correlação de Spearman entre as variáveis
# Não há suposição de distribuição normal das variáveis
fig, ax = plt.subplots(figsize=(20,16))
corr_spearman = sns.heatmap(ds_completo.corr(method='spearman'), annot=True, linewidths=.5)


# In[ ]:


## Observa-se uma grande correlação entre as variáveis Rendimento Tributável e 'Base de Calculo (RTL)' e 'Imposto Devido'
## Também há uma grande correlação entre Quantidade de Declarantes e População
# Essa correlação é esperada


# In[ ]:


### Redução de Componentes utilizando o conhecimento do especialista ###
## Iremos realizar a combinação de algumas variáveis, de uma maneira padronizada na RFB


# In[ ]:


ds_completo_esp=[]
ds_completo_esp=pd.DataFrame(ds_completo_esp)
ds_completo_esp['Total de Rendimentos'] = ds_completo['Rendim. Tribut.']+ds_completo['Rendim. Tribut. Exclus.']+ds_completo['Rendim. Isentos']
ds_completo_esp['Deduções e Pagamentos'] = ds_completo['D - Contrib. Previd.']+ds_completo['D - Dependentes']+ds_completo['D - Instrucao']+ds_completo['D - Medicas']+ds_completo['D - Livro Caixa']+ds_completo['D - Pensao Aliment']+ds_completo['Desc. Padrao']+ds_completo['Imposto Pago']
ds_completo_esp['Imposto Devido'] = ds_completo['Imposto Devido']
ds_completo_esp['Patrimônio'] = ds_completo['Bens e Direitos']-ds_completo['Dividas e Onus']
ds_completo_esp['PIB per capita'] = ds_completo['PIB per capita']
ds_completo_esp['POPULAÇÃO ESTIMADA'] = ds_completo['POPULAÇÃO ESTIMADA']


# In[ ]:


ds_completo_esp.head()


# In[ ]:


ds_completo_esp.info()


# In[ ]:


### Análise exploratória dos dados após a redução dos atributos ###


# In[ ]:


ds_completo_esp.describe()


# In[ ]:


## Imprimindo os Boxplots e a Distribuição de cada coluna

for valor in ds_completo_esp.columns:
  figure = plt.figure()
  nome = str(valor)  
  figure, axs = plt.subplots(1,2, figsize=(15,8))
  sns.boxplot(y=ds_completo_esp[nome], ax=axs[0]) 
  sns.distplot(ds_completo_esp[nome], ax=axs[1])
  plt.show()  
  figure.savefig('boxplot e distribuição'+ " " + nome + '.png' )


# In[ ]:


## Matriz de correlação de Pearson entre as variáveis
fig, ax = plt.subplots(figsize=(20,16))
corr_pearson = sns.heatmap(ds_completo_esp.corr(method='pearson'), annot=True, linewidths=.5)


# In[ ]:


## Matriz de correlação de Spearman entre as variáveis
fig, ax = plt.subplots(figsize=(20,16))
corr_spearman = sns.heatmap(ds_completo_esp.corr(method='spearman'), annot=True, linewidths=.5)


# In[ ]:


### Execução dos algoritmos de Agrupamento ###
from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from math import sqrt


# In[ ]:


## Transformando os dados em uma matriz

X = ds_completo_esp.values

## Normalizando os dados 
## Como os dados não possuem distribuição normal, será utilizado o MinMaxScaler
normalizador = MinMaxScaler()
Xnorm = normalizador.fit_transform(X)


# In[ ]:


#### K-Means ######

lista_inertia = []
lista_silhueta = []

lista_inertianorm = []
lista_silhuetanorm = []

for i in range(2,10):
    km = KMeans(n_clusters=i, random_state=10).fit(X)
    lista_inertia.append(km.inertia_) 
    lista_silhueta.append(metrics.silhouette_score(X, km.labels_, metric='euclidean'))
        

    kmnorm = KMeans(n_clusters=i, random_state=10).fit(Xnorm)
    lista_inertianorm.append(kmnorm.inertia_) 
    lista_silhuetanorm.append(metrics.silhouette_score(Xnorm, kmnorm.labels_, metric='euclidean'))
    
fig, ax = plt.subplots(2, 2, figsize = (15, 8))
ax[0,0].plot(np.arange(2,10), lista_inertia, 'bx-')
ax[0,0].set_title('INERTIA Dados não normalizados')
ax[0,1].plot(np.arange(2,10), lista_silhueta, 'bx-')
ax[0,1].set_title('SILHOUETTE Dados não normalizados')

ax[1,0].plot(np.arange(2,10), lista_inertianorm, 'bx-')
ax[1,0].set_title('INERTIA Dados normalizados')
ax[1,1].plot(np.arange(2,10), lista_silhuetanorm, 'bx-')
ax[1,1].set_title('SILHOUETTE Dados normalizados')

plt.show()


# In[ ]:


## Calculando o número ótimo dos clusters 
## Utilizando a métrica inertia, qual o ponto mais distante da linha que liga os pontos extremos da curva 
## (2 e 10 clusters)
def numero_otimo_clusters(inertia):
    x1, y1 = 2, inertia[0]
    x2, y2 = 10, inertia[len(inertia)-1]

    distances = []
    for i in range(len(inertia)):
        x0 = i+2
        y0 = inertia[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2
n_clusters = numero_otimo_clusters(lista_inertia)
n_clusters_norm = numero_otimo_clusters(lista_inertianorm)
print("número ótimo de clusters dados não normalizados : %d" % n_clusters)
print("número ótimo de clusters dados normalizados : %d" % n_clusters_norm)


# In[ ]:


## Melhor resultado para o modelo para os dados não normalizados 

km_modelo = KMeans(n_clusters=n_clusters, random_state=10).fit(X)


# In[ ]:


## Predizendo os clusters
kmeans_clusters_esp= []
kmeans_clusters_esp = pd.DataFrame(kmeans_clusters_esp)
kmeans_clusters_esp['clusters']=km_modelo.fit_predict(X)


# In[ ]:


## Tamanho dos clusters
kmeans_clusters_esp.value_counts()


# In[ ]:


## Gerando os clusters
kmeans_clusters_esp.index = ds_completo_esp.index
cluster0_esp = kmeans_clusters_esp[kmeans_clusters_esp['clusters']==0]
cluster1_esp = kmeans_clusters_esp[kmeans_clusters_esp['clusters']==1]
cluster2_esp = kmeans_clusters_esp[kmeans_clusters_esp['clusters']==2]
cluster3_esp = kmeans_clusters_esp[kmeans_clusters_esp['clusters']==3]
cluster4_esp = kmeans_clusters_esp[kmeans_clusters_esp['clusters']==4]


# In[ ]:


cluster2_esp.head()


# In[ ]:


cluster1_esp.head()


# In[ ]:


cluster4_esp.head(22)


# In[ ]:


#### Visões sobre os clusters ###


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('Patrimônio')
plt.ylabel('Total de Rendimentos')
plt.scatter(ds_completo_esp['Patrimônio'], ds_completo_esp['Total de Rendimentos'], c= kmeans_clusters_esp['clusters'],  cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('Patrimônio')
plt.ylabel('POPULAÇÃO ESTIMADA')
plt.scatter(ds_completo_esp['Patrimônio'], ds_completo_esp['POPULAÇÃO ESTIMADA'], c= kmeans_clusters_esp['clusters'], cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('Total de Rendimentos')
plt.ylabel('Deduções e Pagamentos')
plt.scatter(ds_completo_esp['Total de Rendimentos'], ds_completo_esp['Deduções e Pagamentos'], c= kmeans_clusters_esp['clusters'], cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('PIB per capita')
plt.ylabel('Imposto Devido')
plt.scatter(ds_completo_esp['PIB per capita'], ds_completo_esp['Imposto Devido'], c= kmeans_clusters_esp['clusters'], cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('PIB per capita')
plt.ylabel('POPULAÇÃO ESTIMADA')
plt.scatter(ds_completo_esp['PIB per capita'], ds_completo_esp['POPULAÇÃO ESTIMADA'], c= kmeans_clusters_esp['clusters'], cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('Patrimônio')
plt.ylabel('Deduções e Pagamentos')
plt.scatter(ds_completo_esp['Patrimônio'], ds_completo_esp['Deduções e Pagamentos'], c= kmeans_clusters_esp['clusters'], cmap='plasma')


# In[ ]:


#### Clusters dos Dados Normalizados ####


# In[ ]:


## Melhor resultado para o modelo para os dados não normalizados

km_modelo_norm = KMeans(n_clusters=n_clusters_norm, random_state=10).fit(Xnorm)


# In[ ]:


## Predizendo os clusters, para utilizar como cores para os gráficos
kmeans_clusters_esp_norm= []
kmeans_clusters_esp_norm = pd.DataFrame(kmeans_clusters_esp_norm)
kmeans_clusters_esp_norm['clusters']=km_modelo_norm.fit_predict(Xnorm)


# In[ ]:


## Tamanho dos clusters
kmeans_clusters_esp_norm.value_counts()


# In[ ]:


## Gerando os clusters
kmeans_clusters_esp_norm.index = ds_completo_esp.index
cluster0_esp_norm = kmeans_clusters_esp_norm[kmeans_clusters_esp_norm['clusters']==0]
cluster1_esp_norm = kmeans_clusters_esp_norm[kmeans_clusters_esp_norm['clusters']==1]
cluster2_esp_norm = kmeans_clusters_esp_norm[kmeans_clusters_esp_norm['clusters']==2]
cluster3_esp_norm = kmeans_clusters_esp_norm[kmeans_clusters_esp_norm['clusters']==3]
cluster4_esp_norm = kmeans_clusters_esp_norm[kmeans_clusters_esp_norm['clusters']==4]
cluster5_esp_norm = kmeans_clusters_esp_norm[kmeans_clusters_esp_norm['clusters']==5]


# In[ ]:


#### Visões sobre os clusters ###


# In[ ]:


## Gráficos de clusters
plt.figure(figsize=(15,5))
plt.xlabel('Patrimônio')
plt.ylabel('Total de Rendimentos')
plt.scatter(ds_completo_esp['Patrimônio'], ds_completo_esp['Total de Rendimentos'], c= kmeans_clusters_esp_norm['clusters'],  cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('PIB per capita')
plt.ylabel('Total de Rendimentos')
plt.scatter(ds_completo_esp['PIB per capita'], ds_completo_esp['Total de Rendimentos'], c= kmeans_clusters_esp_norm['clusters'],  cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('PIB per capita')
plt.ylabel('Patrimônio')
plt.scatter(ds_completo_esp['PIB per capita'], ds_completo_esp['Patrimônio'], c= kmeans_clusters_esp_norm['clusters'], cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('Total de Rendimentos')
plt.ylabel('Deduções e Pagamentos')
plt.scatter(ds_completo_esp['Total de Rendimentos'], ds_completo_esp['Deduções e Pagamentos'], c= kmeans_clusters_esp_norm['clusters'], cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('Patrimônio')
plt.ylabel('Deduções e Pagamentos')
plt.scatter(ds_completo_esp['Patrimônio'], ds_completo_esp['Deduções e Pagamentos'], c= kmeans_clusters_esp_norm['clusters'], cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('PIB per capita')
plt.ylabel('Deduções e Pagamentos')
plt.scatter(ds_completo_esp['PIB per capita'], ds_completo_esp['Deduções e Pagamentos'], c= kmeans_clusters_esp_norm['clusters'], cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('Patrimônio')
plt.ylabel('POPULAÇÃO ESTIMADA')
plt.scatter(ds_completo_esp['Patrimônio'], ds_completo_esp['POPULAÇÃO ESTIMADA'], c= kmeans_clusters_esp_norm['clusters'], cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('PIB per capita')
plt.ylabel('Imposto devido')
plt.scatter(ds_completo_esp['PIB per capita'], ds_completo_esp['Imposto Devido'], c= kmeans_clusters_esp_norm['clusters'], cmap='plasma')


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel('Deduções e Pagamentos')
plt.ylabel('Imposto devido')
plt.scatter(ds_completo_esp['Deduções e Pagamentos'], ds_completo_esp['Imposto Devido'], c= kmeans_clusters_esp_norm['clusters'], cmap='plasma')


# In[ ]:


### PCA - Redução de Componentes ###


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


## Dados normalizados
pca_norm = PCA().fit(Xnorm)


# In[ ]:


plt.plot(np.cumsum(pca_norm.explained_variance_ratio_), 'bx-')
plt.grid(True)
plt.xlabel('número de componentes')
plt.ylabel('variância acumulada')
print(pca_norm.explained_variance_ratio_)
print(np.cumsum(pca_norm.explained_variance_ratio_))


# In[ ]:


# Apenas 2 componentes já possuem 0.87 da variância acumulada


# In[ ]:


# Gerando as duas componentes do PCA
pca_norm = PCA(n_components=2)
pca_norm_components = pca_norm.fit_transform(Xnorm)


# In[ ]:


# Transformando em um Dataframe
df_pca_norm_components = pd.DataFrame(pca_norm_components)


# In[ ]:


## Transformando os pesos dos componentes em um DataFrame
df_pca_norm_components_pesos = pd.DataFrame(pca_norm.components_)
df_pca_norm_components_pesos.columns = ds_completo_esp.columns
df_pca_norm_components_pesos.head()


# In[ ]:


## Como foi possível observar antes da redução do PCA,
## Deduções e Pagamentos, Imposto Devido  e PIB per capita tiveram os maiores pesos


# In[ ]:


df_pca_norm_components_pesos.to_excel('df_pca_norm_components_pesos.xlsx')


# In[ ]:


#### K-Means - PCA ######


lista_inertianorm = []
lista_silhuetanorm = []

for i in range(2,10):
    km_norm = KMeans(n_clusters=i, random_state=10).fit(pca_norm_components)
    lista_inertianorm.append(km_norm.inertia_) 
    lista_silhuetanorm.append(metrics.silhouette_score(pca_norm_components, km_norm.labels_, metric='euclidean'))
   

fig, ax = plt.subplots(1, 2, figsize = (15, 8))

ax[0].plot(np.arange(2,10), lista_inertianorm, 'bx-')
ax[0].set_title('INERTIA Dados normalizados')

ax[1].plot(np.arange(2,10), lista_silhuetanorm, 'bx-')
ax[1].set_title('SILHOUETTE Dados normalizados')

plt.show()


# In[ ]:


## Calculando o número ótimo dos clusters 
## Utilizando a métrica inertia, qual o ponto mais distante da linha que liga os pontos extremos da curva 
## (2 e 10 clusters)
def numero_otimo_clusters(inertia):
    x1, y1 = 2, inertia[0]
    x2, y2 = 10, inertia[len(inertia)-1]

    distances = []
    for i in range(len(inertia)):
        x0 = i+2
        y0 = inertia[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2

n_clusters_pca_norm = numero_otimo_clusters(lista_inertianorm)
print("número ótimo de clusters dados normalizados : %d" % n_clusters_pca_norm)


# In[ ]:


## Melhor resultado, dados normalizados

km_modelo_pca_norm = KMeans(n_clusters=n_clusters_pca_norm, random_state=10).fit(pca_norm_components)


# In[ ]:


## Predizendo os clusters, para utilizar como cores para os gráficos
kmeans_clusters_pca_norm = []
kmeans_clusters_pca_norm = pd.DataFrame(kmeans_clusters_pca_norm)
kmeans_clusters_pca_norm['clusters']=km_modelo_pca_norm.fit_predict(pca_norm_components)


# In[ ]:


## Tamanho dos clusters
kmeans_clusters_pca_norm.value_counts()


# In[ ]:


#### Visões sobre os clusters ###


# In[ ]:


## Gráficos de clusters
## Os centroides foram marcados com o número do respectivo cluster
plt.figure(figsize=(15,5))
plt.xlabel('PCA1')
plt.ylabel('PCA0')
plt.scatter( df_pca_norm_components[1], df_pca_norm_components[0], c= kmeans_clusters_pca_norm['clusters'],cmap='rainbow')
plt.scatter(km_modelo_pca_norm.cluster_centers_[0,1] , km_modelo_pca_norm.cluster_centers_[0,0], color='black', marker='$0$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[1,1] , km_modelo_pca_norm.cluster_centers_[1,0], color='black', marker='$1$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[2,1] , km_modelo_pca_norm.cluster_centers_[2,0], color='black', marker='$2$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[3,1] , km_modelo_pca_norm.cluster_centers_[3,0], color='black', marker='$3$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[4,1] , km_modelo_pca_norm.cluster_centers_[4,0], color='black', marker='$4$')


# In[ ]:


## Gráficos de clusters
## Versão com uma transparência de 25%

plt.figure(figsize=(15,5))
plt.xlabel('PCA1')
plt.ylabel('PCA0')
plt.scatter( df_pca_norm_components[1], df_pca_norm_components[0], c= kmeans_clusters_pca_norm['clusters'],cmap='rainbow', alpha = 0.25)
plt.scatter(km_modelo_pca_norm.cluster_centers_[0,1] , km_modelo_pca_norm.cluster_centers_[0,0], color='black', marker='$0$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[1,1] , km_modelo_pca_norm.cluster_centers_[1,0], color='black', marker='$1$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[2,1] , km_modelo_pca_norm.cluster_centers_[2,0], color='black', marker='$2$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[3,1] , km_modelo_pca_norm.cluster_centers_[3,0], color='black', marker='$3$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[4,1] , km_modelo_pca_norm.cluster_centers_[4,0], color='black', marker='$4$')


# In[ ]:


# Observa-se que os clusters ficaram bem definidos e este modelo será utilizado para análise dos resultados


# In[ ]:


# Incluindo a métrica de "Carga Tributária" no dataset original, para auxiliar na análise

ds_completo_esp['Carga Tributária'] = 100*(ds_completo_esp['Imposto Devido']*ds_completo['Qtde Declarantes'])/(ds_completo_esp['PIB per capita']*ds_completo_esp['POPULAÇÃO ESTIMADA'])


# In[ ]:


ds_completo_esp.describe()


# In[ ]:


ds_completo_esp.describe().to_excel('ds_completo_esp.describe.xlsx')


# In[ ]:


## Imprimindo os Boxplots e a Distribuição da "Carga Tributária"

figure = plt.figure()
figure, axs = plt.subplots(1,2, figsize=(15,8))
sns.boxplot(y=ds_completo_esp['Carga Tributária'], ax=axs[0]) 
sns.distplot(ds_completo_esp['Carga Tributária'], ax=axs[1])
plt.show()  
figure.savefig('boxplot e distribuição'+ " " + 'Carga Tributária' + '.png' )


# In[ ]:


## Criando uma coluna com o Porte populacional dos municípios
## para auxiliar na análise
import sys
ds_completo_esp['Porte Município']=pd.cut(ds_completo_esp['POPULAÇÃO ESTIMADA'], bins=[0, 20000, 100000, sys.maxsize],labels=['pequeno porte','médio porte','grande porte'])


# In[ ]:


ds_completo_esp['Porte Município'].value_counts()


# In[ ]:


## Recuperando a informação da Atividade com maior valor adicionado ao PIB
## para auxiliar na análise

ds_IBGE3 = ds_IBGE[['Atividade com maior valor adicionado bruto']]


# In[ ]:


ds_completo_esp = ds_completo_esp.merge(ds_IBGE3, left_index=True, right_index=True)


# In[ ]:


ds_completo_esp['Atividade com maior valor adicionado bruto'].value_counts()


# In[ ]:


ds_completo_esp.to_excel('ds_completo_esp.xlsx')


# In[ ]:


## Gerando os clusters
kmeans_clusters_pca_norm.index = ds_completo_esp.index
cluster0_pca_norm = kmeans_clusters_pca_norm[kmeans_clusters_pca_norm['clusters']==0]
cluster1_pca_norm = kmeans_clusters_pca_norm[kmeans_clusters_pca_norm['clusters']==1]
cluster2_pca_norm = kmeans_clusters_pca_norm[kmeans_clusters_pca_norm['clusters']==2]
cluster3_pca_norm = kmeans_clusters_pca_norm[kmeans_clusters_pca_norm['clusters']==3]
cluster4_pca_norm = kmeans_clusters_pca_norm[kmeans_clusters_pca_norm['clusters']==4]


# In[ ]:


## Incluindo os dados originais nos clusters, para análise

cluster0_pca_norm = cluster0_pca_norm.merge(ds_completo_esp, left_index=True, right_index=True)
cluster1_pca_norm = cluster1_pca_norm.merge(ds_completo_esp, left_index=True, right_index=True)
cluster2_pca_norm = cluster2_pca_norm.merge(ds_completo_esp, left_index=True, right_index=True)
cluster3_pca_norm = cluster3_pca_norm.merge(ds_completo_esp, left_index=True, right_index=True)
cluster4_pca_norm = cluster4_pca_norm.merge(ds_completo_esp, left_index=True, right_index=True)


# In[ ]:


## Salvando os clusters para Excel

cluster0_pca_norm.to_excel('cluster0_pca_norm.xlsx')
cluster1_pca_norm.to_excel('cluster1_pca_norm.xlsx')
cluster2_pca_norm.to_excel('cluster2_pca_norm.xlsx')
cluster3_pca_norm.to_excel('cluster3_pca_norm.xlsx')
cluster4_pca_norm.to_excel('cluster4_pca_norm.xlsx')


# In[ ]:


## Salvando os sumários estatísticos dos clusters para Excel

cluster0_pca_norm.describe().to_excel('cluster0_pca_norm_describe.xlsx')
cluster1_pca_norm.describe().to_excel('cluster1_pca_norm_describe.xlsx')
cluster2_pca_norm.describe().to_excel('cluster2_pca_norm_describe.xlsx')
cluster3_pca_norm.describe().to_excel('cluster3_pca_norm_describe.xlsx')
cluster4_pca_norm.describe().to_excel('cluster4_pca_norm_describe.xlsx')


# In[ ]:


## Gerando as informações de cada cluster para auxiliar na análise
## Sumário estatístico, distribuição do Porte populacional e da Atividade econômica mais relevante


# In[ ]:


# Cluster 0
cluster0_pca_norm.describe()


# In[ ]:


cluster0_pca_norm['Porte Município'].value_counts()


# In[ ]:


cluster0_pca_norm['Atividade com maior valor adicionado bruto'].value_counts()


# In[ ]:


# Cluster 1
cluster1_pca_norm.describe()


# In[ ]:


cluster1_pca_norm['Porte Município'].value_counts()


# In[ ]:


cluster1_pca_norm['Atividade com maior valor adicionado bruto'].value_counts()


# In[ ]:


# Cluster 2
cluster2_pca_norm.describe()


# In[ ]:


cluster2_pca_norm['Porte Município'].value_counts()


# In[ ]:


cluster2_pca_norm['Atividade com maior valor adicionado bruto'].value_counts()


# In[ ]:


# Cluster 3
cluster3_pca_norm.describe()


# In[ ]:


cluster3_pca_norm['Porte Município'].value_counts()


# In[ ]:


cluster3_pca_norm['Atividade com maior valor adicionado bruto'].value_counts()


# In[ ]:


# Cluster 4
cluster4_pca_norm.describe()


# In[ ]:


cluster4_pca_norm['Porte Município'].value_counts()


# In[ ]:


cluster4_pca_norm['Atividade com maior valor adicionado bruto'].value_counts()


# In[ ]:


## Calculando a distância dos pontos para os Centróides ##

distancias_centroides = km_modelo_pca_norm.transform(pca_norm_components)
distancias_centroides = pd.DataFrame(distancias_centroides)
distancias_centroides.index = ds_completo_esp.index 
distancias_centroides.columns = ['cluster0', 'cluster1', 'cluster2', 'cluster3', 'cluster4']
distancias_centroides.head() 


# In[ ]:


## Calculando o somatório das distâncias de cada cluster, para avaliar o nível de compactação


# In[ ]:


# Cluster 0
distancias_centroides['cluster0'].sum()


# In[ ]:


# Cluster 1
distancias_centroides['cluster1'].sum()


# In[ ]:


# Cluster 2
distancias_centroides['cluster2'].sum()


# In[ ]:


# Cluster 3
distancias_centroides['cluster3'].sum()


# In[ ]:


# Cluster 4
distancias_centroides['cluster4'].sum()


# In[ ]:


## Calculando a métrica Silhouette para cada cluster


# In[ ]:


# Cluster 0
metrics.silhouette_score(pca_norm_components, km_modelo_pca_norm.labels_ == 0, metric='euclidean')


# In[ ]:


# Cluster 1
metrics.silhouette_score(pca_norm_components, km_modelo_pca_norm.labels_ == 1, metric='euclidean')


# In[ ]:


# Cluster 2
metrics.silhouette_score(pca_norm_components, km_modelo_pca_norm.labels_ == 2, metric='euclidean')


# In[ ]:


# Cluster 3
metrics.silhouette_score(pca_norm_components, km_modelo_pca_norm.labels_ == 3, metric='euclidean')


# In[ ]:


# Cluster 4
metrics.silhouette_score(pca_norm_components, km_modelo_pca_norm.labels_ == 4, metric='euclidean')


# In[ ]:


## Criando dataframes para as distâncias de cada cluster ##
## Para a extração dos Outliers ##
distancias_centroides = distancias_centroides.merge(kmeans_clusters_pca_norm, left_index=True  , right_index=True)
dist_cluster0_pca_norm = distancias_centroides[distancias_centroides['clusters']==0]
dist_cluster1_pca_norm = distancias_centroides[distancias_centroides['clusters']==1]
dist_cluster2_pca_norm = distancias_centroides[distancias_centroides['clusters']==2]
dist_cluster3_pca_norm = distancias_centroides[distancias_centroides['clusters']==3]
dist_cluster4_pca_norm = distancias_centroides[distancias_centroides['clusters']==4]


# In[ ]:


distancias_centroides.to_excel('distancias_centroides.xlsx')


# In[ ]:


## Calculando o percentil 99, para extrair os Outliers dos clusters
distancias_centroides_describe0 = dist_cluster0_pca_norm.describe(percentiles=[.99])
distancias_centroides_describe1 = dist_cluster1_pca_norm.describe(percentiles=[.99])
distancias_centroides_describe2 = dist_cluster2_pca_norm.describe(percentiles=[.99])
distancias_centroides_describe3 = dist_cluster3_pca_norm.describe(percentiles=[.99])
distancias_centroides_describe4 = dist_cluster4_pca_norm.describe(percentiles=[.99])


# In[ ]:


limite_max_cluster0 = distancias_centroides_describe0.loc['99%', 'cluster0']
limite_max_cluster1 = distancias_centroides_describe1.loc['99%', 'cluster1']
limite_max_cluster2 = distancias_centroides_describe2.loc['99%', 'cluster2']
limite_max_cluster3 = distancias_centroides_describe3.loc['99%', 'cluster3']
limite_max_cluster4 = distancias_centroides_describe4.loc['99%', 'cluster4']


# In[ ]:


dist_cluster0_pca_norm_max = dist_cluster0_pca_norm[dist_cluster0_pca_norm['cluster0'] >= limite_max_cluster0 ]
dist_cluster0_pca_norm_max.info()


# In[ ]:


dist_cluster1_pca_norm_max = dist_cluster1_pca_norm[dist_cluster1_pca_norm['cluster1'] >= limite_max_cluster1 ]
dist_cluster1_pca_norm_max.info()


# In[ ]:


dist_cluster2_pca_norm_max = dist_cluster2_pca_norm[dist_cluster2_pca_norm['cluster2'] >= limite_max_cluster2 ]
dist_cluster2_pca_norm_max.info()


# In[ ]:


dist_cluster3_pca_norm_max = dist_cluster3_pca_norm[dist_cluster3_pca_norm['cluster3'] >= limite_max_cluster3 ]
dist_cluster3_pca_norm_max.info()


# In[ ]:


dist_cluster4_pca_norm_max = dist_cluster4_pca_norm[dist_cluster4_pca_norm['cluster4'] >= limite_max_cluster4 ]
dist_cluster4_pca_norm_max.info()


# In[ ]:


## Concatenando os Outliers em um único Dataframe

dist_cluster_pca_norm_max = []
dist_cluster_pca_norm_max = pd.DataFrame(dist_cluster_pca_norm_max)
dist_cluster_pca_norm_max = pd.concat([dist_cluster0_pca_norm_max, dist_cluster1_pca_norm_max, dist_cluster2_pca_norm_max ,dist_cluster3_pca_norm_max ,dist_cluster4_pca_norm_max])
dist_cluster_pca_norm_max.info()
dist_cluster_pca_norm_max.head()


# In[ ]:


## Incluindo os valores originais nos Outliers para análise
dist_cluster_pca_norm_max_merge = dist_cluster_pca_norm_max.merge(ds_completo_esp, left_index=True  , right_index=True, how ='left')
dist_cluster_pca_norm_max_merge.info()
dist_cluster_pca_norm_max_merge.head()


# In[ ]:


dist_cluster_pca_norm_max_merge.to_excel('dist_cluster_pca_norm_max_merge.xlsx')


# In[ ]:


dist_cluster_pca_norm_max_merge.describe().to_excel('dist_cluster_pca_norm_max_merge_describe.xlsx')


# In[ ]:


## Gráficos de clusters com os Outliers
# Cluster 0

df_pca_norm_components2 = df_pca_norm_components
df_pca_norm_components2.index = ds_completo_esp.index

plt.figure(figsize=(15,5))
plt.xlabel('PCA1')
plt.ylabel('PCA0')
plt.scatter( df_pca_norm_components[1], df_pca_norm_components[0], c= kmeans_clusters_pca_norm['clusters'],cmap='rainbow', alpha=1)
plt.scatter(km_modelo_pca_norm.cluster_centers_[0,1] , km_modelo_pca_norm.cluster_centers_[0,0], color='black', marker='$0$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[1,1] , km_modelo_pca_norm.cluster_centers_[1,0], color='black', marker='$1$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[2,1] , km_modelo_pca_norm.cluster_centers_[2,0], color='black', marker='$2$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[3,1] , km_modelo_pca_norm.cluster_centers_[3,0], color='black', marker='$3$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[4,1] , km_modelo_pca_norm.cluster_centers_[4,0], color='black', marker='$4$')
plt.scatter(df_pca_norm_components2.loc[dist_cluster0_pca_norm_max.index, 1], df_pca_norm_components2.loc[dist_cluster0_pca_norm_max.index, 0], color='black', marker='X')


# In[ ]:


## Gráficos de clusters com os Outliers
# Cluster 1

plt.figure(figsize=(15,5))
plt.xlabel('PCA1')
plt.ylabel('PCA0')
plt.scatter( df_pca_norm_components[1], df_pca_norm_components[0], c= kmeans_clusters_pca_norm['clusters'],cmap='rainbow', alpha=1)
plt.scatter(km_modelo_pca_norm.cluster_centers_[0,1] , km_modelo_pca_norm.cluster_centers_[0,0], color='black', marker='$0$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[1,1] , km_modelo_pca_norm.cluster_centers_[1,0], color='black', marker='$1$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[2,1] , km_modelo_pca_norm.cluster_centers_[2,0], color='black', marker='$2$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[3,1] , km_modelo_pca_norm.cluster_centers_[3,0], color='black', marker='$3$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[4,1] , km_modelo_pca_norm.cluster_centers_[4,0], color='black', marker='$4$')
plt.scatter(df_pca_norm_components2.loc[dist_cluster1_pca_norm_max.index, 1], df_pca_norm_components2.loc[dist_cluster1_pca_norm_max.index, 0], color='black', marker='X')


# In[ ]:


## Gráficos de clusters com os Outliers
# Cluster 2

plt.figure(figsize=(15,5))
plt.xlabel('PCA1')
plt.ylabel('PCA0')
plt.scatter( df_pca_norm_components[1], df_pca_norm_components[0], c= kmeans_clusters_pca_norm['clusters'],cmap='rainbow', alpha=1)
plt.scatter(km_modelo_pca_norm.cluster_centers_[0,1] , km_modelo_pca_norm.cluster_centers_[0,0], color='black', marker='$0$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[1,1] , km_modelo_pca_norm.cluster_centers_[1,0], color='black', marker='$1$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[2,1] , km_modelo_pca_norm.cluster_centers_[2,0], color='black', marker='$2$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[3,1] , km_modelo_pca_norm.cluster_centers_[3,0], color='black', marker='$3$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[4,1] , km_modelo_pca_norm.cluster_centers_[4,0], color='black', marker='$4$')
plt.scatter(df_pca_norm_components2.loc[dist_cluster2_pca_norm_max.index, 1], df_pca_norm_components2.loc[dist_cluster2_pca_norm_max.index, 0], color='black', marker='X')


# In[ ]:


## Gráficos de clusters com os Outliers
# Cluster 3

plt.figure(figsize=(15,5))
plt.xlabel('PCA1')
plt.ylabel('PCA0')
plt.scatter( df_pca_norm_components[1], df_pca_norm_components[0], c= kmeans_clusters_pca_norm['clusters'],cmap='rainbow', alpha=1)
plt.scatter(km_modelo_pca_norm.cluster_centers_[0,1] , km_modelo_pca_norm.cluster_centers_[0,0], color='black', marker='$0$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[1,1] , km_modelo_pca_norm.cluster_centers_[1,0], color='black', marker='$1$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[2,1] , km_modelo_pca_norm.cluster_centers_[2,0], color='black', marker='$2$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[3,1] , km_modelo_pca_norm.cluster_centers_[3,0], color='black', marker='$3$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[4,1] , km_modelo_pca_norm.cluster_centers_[4,0], color='black', marker='$4$')
plt.scatter(df_pca_norm_components2.loc[dist_cluster3_pca_norm_max.index, 1], df_pca_norm_components2.loc[dist_cluster3_pca_norm_max.index, 0], color='black', marker='X')


# In[ ]:


## Gráficos de clusters com os Outliers
# Cluster 4

plt.figure(figsize=(15,5))
plt.xlabel('PCA1')
plt.ylabel('PCA0')
plt.scatter( df_pca_norm_components[1], df_pca_norm_components[0], c= kmeans_clusters_pca_norm['clusters'],cmap='rainbow', alpha=1)
plt.scatter(km_modelo_pca_norm.cluster_centers_[0,1] , km_modelo_pca_norm.cluster_centers_[0,0], color='black', marker='$0$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[1,1] , km_modelo_pca_norm.cluster_centers_[1,0], color='black', marker='$1$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[2,1] , km_modelo_pca_norm.cluster_centers_[2,0], color='black', marker='$2$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[3,1] , km_modelo_pca_norm.cluster_centers_[3,0], color='black', marker='$3$')
plt.scatter(km_modelo_pca_norm.cluster_centers_[4,1] , km_modelo_pca_norm.cluster_centers_[4,0], color='black', marker='$4$')
plt.scatter(df_pca_norm_components2.loc[dist_cluster4_pca_norm_max.index, 1], df_pca_norm_components2.loc[dist_cluster4_pca_norm_max.index, 0], color='black', marker='X')


# In[ ]:


### FIM DO SCRIPT ###

