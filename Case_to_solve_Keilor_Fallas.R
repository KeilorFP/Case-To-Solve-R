#KEILOR FALLAS PRADO
#CASE TO SOLVE R


# Instalar las librerías necesarias
install.packages("xgboost")
install.packages("caret")
install.packages("dplyr")

# Luego cargamos las librerías necesarias
library(xgboost)
library(caret)
library(dplyr)

# Leemos los datos desde la ubicacion de donde tenemos el archivo csv
df_wines <- read.csv("C:/Users/kfall/Desktop/Case to solve R/df_wines.csv", stringsAsFactors = TRUE)

# Ver el resumen de los datos
str(df_wines)
summary(df_wines)
# En el resumen logramos ver que hay vinos de diferentes paises, el rating va de 2.2 hasta 4.7.
#6. En el precio vemos que la mayoría de los vinos cuestan entre $9.77 y $28.39


# Comprobar valores nulos
df_wines %>% summarise_all(~sum(is.na(.)))
# Por lo que no encontramos valores nulos en ninguna de las columnas

# Visualización inicial de la distribución de precios
hist(df_wines$Price, main="Distribución de Precios", xlab="Precio", col="lightblue", breaks=30)
# Encontramos que los vinos mas comprados se encuentran entre 5 a 20 euros y entre mas alto el precio con menor frecuencia son comprados.

# Transformación logarítmica en la variable Price, porque en la variable price queremos hacer los valores mas simetricos para obtner un mejor modelo y esta conclusion lo vemos en el histograma.
df_wines$Log_Price <- log(df_wines$Price + 1) 

# 1. Análisis de varianza (ANOVA)
anova_resultado <- aov(Log_Price ~ wine_type, data = df_wines)
summary(anova_resultado)

# se realizo la prueba ANOVA y tuvo resultados significativos por lo que se realizo la prueba tukey.
tukey_resultado <- TukeyHSD(anova_resultado)
summary(tukey_resultado)

#Espumoso vs Blanco: El vino espumoso es más caro que el blanco (+0.313)
#Rosado vs Blanco: El vino rosado es más barato que el blanco (−0.338)
#Tinto vs Blanco: El vino tinto es más caro que el blanco (+0.256)
#Rosado vs Espumoso: El vino rosado es significativamente más barato que el espumoso (−0.652)
#Tinto vs Rosado: El vino tinto es significativamente más caro que el rosado (+0.595)
#Tinto vs Espumoso: No hay suficiente evidencia para decir que hay una diferencia de precio significativa entre los vinos tintos y espumosos

# Imprimimos los resultados
print(tukey_resultado)
plot(tukey_resultado)

# Despues de ver los resultados podemos concluir que los vinos tintos y espumosos tienden a ser más caros, mientras que los vinos rosados son los más baratos. Sin embargo, la diferencia entre tinto y espumoso no podemos deducirlo

# Comparación de precios por tipo de vino
boxplot(Log_Price ~ wine_type, data=df_wines, main="Precios logarítmicos por Tipo de Vino", col=c("red", "blue"))
# En boxplot encontramos algunos outliers en los precios, pero se utiliza el modelo de xgboost para controlar estos outliers y obtener mejor prediccion.

# Convertir variables categóricas a factores
categorical_columns <- c('Name', 'Country', 'Region', 'Winery', 'wine_type')
df_wines[categorical_columns] <- lapply(df_wines[categorical_columns], factor)


# Realizar One-Hot Encoding de las variables categóricas
df_wines <- cbind(df_wines, model.matrix(~Name + Country + Region + Winery + wine_type - 1, data=df_wines))
df_wines <- df_wines[, !(names(df_wines) %in% categorical_columns)]  # Aqui se eliminan las columnas originales de texto

# Escalar las variables numéricas, aqui lo utilizamos para obtener un rango similar en los datos.
numerical_columns <- c('Rating', 'NumberOfRatings', 'Price', 'Year1')  # Ajusta según tus columnas numéricas
df_wines[numerical_columns] <- scale(df_wines[numerical_columns])

# Dividir los datos en entrenamiento y prueba, aqui vamos a tener un 80% de train y 20% de test
set.seed(42) 
train_index <- createDataPartition(df_wines$Price, p=0.8, list=FALSE)  
train_data <- df_wines[train_index, ]
test_data <- df_wines[-train_index, ]

# Variables independientes (X) y dependiente (Y), aqui lo hacemos en matrix para que el modelo xgboost funcione correctamente.
X_train <- as.matrix(train_data %>% select(-Price))
y_train <- train_data$Price
X_test <- as.matrix(test_data %>% select(-Price))
y_test <- test_data$Price


# Convertir los datos a DMatrix para obtener una mejor optmizacion en la memoria
train_matrix <- xgb.DMatrix(data = X_train, label = y_train)
test_matrix <- xgb.DMatrix(data = X_test, label = y_test)

# Aqui definimos los parametros
params <- list(
  objective = "reg:squarederror", 
  booster = "gbtree",  # Árboles de decisión
  eta = 0.1,  # Tasa de aprendizaje
  max_depth = 6,  # Profundidad máxima de los árboles
  colsample_bytree = 0.8,  # Muestra de características
  subsample = 0.8  # Muestra de observaciones
)

# Procedemos al entrenamiento del modelo XGBoost
xgb_model <- xgboost(
  params = params,
  data = train_matrix,
  nrounds = 100,  # Número de iteraciones, aqui lo hice con 60 pero obtuve mejor resultados con 100
  verbose = 1
)

# Ver resumen del modelo
print(xgb_model)

# Predicciones con el modelo entrenado
y_pred <- predict(xgb_model, test_matrix)

# Calcular R², Aqui obtuve R²:  0.9996748 indica que el modelo es capaz de explicar el 99.97% de la variabilidad en los datos y el modelo esta adecuado para proceguir.
r2 <- cor(y_test, y_pred)^2
cat("R²: ", r2, "\n")

# Calcular RMSE (Error cuadrático medio), aqui obtuve RMSE:  0.01807945 por lo que es un bajo de error muy bajo lo que indica que las predicciones del modelo están muy cerca de los valores reales
rmse <- sqrt(mean((y_test - y_pred)^2))
cat("RMSE: ", rmse, "\n")


# Definimos el espacio de búsqueda aleatoria 
tune_grid_random <- list(
  nrounds = c(50, 100, 150),    # Número de iteraciones
  max_depth = c(3, 6, 9, 12),   # Profundidad máxima del árbol
  eta = c(0.01, 0.1, 0.3),     # Tasa de aprendizaje
  gamma = c(0, 1, 5),           # Reducción de la complejidad
  colsample_bytree = c(0.6, 0.8, 1),   # Fracción de columnas a muestrear por árbol
  min_child_weight = c(1, 5, 10), # Peso mínimo de los nodos
  subsample = c(0.6, 0.8, 1)    # Fracción de datos a usar en cada árbol
)

# Usamos train para la búsqueda aleatoria de hiperparámetros
set.seed(42)
xgb_tuned_random <- train(
  x = X_train, y = y_train,
  method = "xgbTree",
  tuneLength = 5,  # Es el número de combinaciones aleatorias a probar, lo hice con 10 pero la duracion fue muy duradera por lo que lo baje a 5
  trControl = trainControl(method = "cv", number = 5),  # Validación cruzada de 5
  tuneGrid = expand.grid(tune_grid_random)
)

# Imprimimos los mejores parámetros encontrados
best_params_random <- xgb_tuned_random$bestTune
print(best_params_random)

#  Y vemos el rendimiento en el conjunto de validación cruzada
print(xgb_tuned_random)


# Conclusiones 
# Intente entrenar el modelo, pero el incoveniente es que demora demasiado tiempo para el entrenamiento por lo caul no logre obtener resultados y saber si el modelo estaba sobre ajustado