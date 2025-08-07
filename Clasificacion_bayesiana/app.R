library(shiny)

ui<-fluidPage(

  titlePanel("Clasificador Bayesiano"),
  p("¡Bienvenido a la App de Clasificación Bayesiana!"),
  p("Este programa utiliza el método de Naive-Bayes para realizar predicciones sobre una variable discreta desconocida llamada clase.
    Para ello, se debe proporcionar una base de datos que disponga de instancias de clase conocida para entrenar al modelo."),
  p("Los archivos adjuntados deben estar en formato .csv. Las variables continuas deben ser de tipo \"numeric\" y las discretas de tipo \"character\"."),
  p("El programa entrenará el modelo con una proporción, a elección del usuario, de los datos de entrenamiento y testeo 
    y devolverá una serie de métricas de validación del algoritmo obtenidas utilizando el resto de los datos."),
  p("Opcionalmente, se puede adjuntar también un archivo de datos de predicción sobre los que se desee obtener una predicción de la clase.
    Este archivo debe de tener la misma configuración que los datos de entrenamiento y testeo, pero sin la columna de la variable clase."),
  sidebarLayout(
    sidebarPanel(
      h4("Parámetros de entrada"),
      textInput("target","Nombre de la variable clase"),
      textInput("dominio","Dominio de la clase (separados por comas):", 
                "a,b,c,..."),
      numericInput("prop_entrenamiento", "Proporción de entrenamiento",0.7,min=0,max=1,step=0.1),
      fileInput("ubicacion","Datos de entrenamiento y testeo. Cargar archivo .csv",accept=".csv"),
      fileInput("ubicacion_prediccion","Datos de predicción. Cargar archivo .csv",accept=".csv"),
      actionButton("ejecutar", "Ejecutar")
    ),
    
    mainPanel(
      h3("Rendimiento del modelo"),
      hr(),
      
      h4("Matriz de Confusión"),
      h5("Filas:Valores Reales. Columnas:Valores Predichos."),
      tableOutput("matriz_confusion"),
      hr(),
      
      h4("Exactitud"),
      textOutput("exactitud"),
      hr(),
      
      h4("Métricas por clase"),
      tableOutput("tabla_metricas"),
      hr(),
      
      h4("Macro F1"),
      textOutput("macro_f1"),
      hr(),
      
      h4("Macro F1 ponderado"),
      textOutput("macro_f1_ponderado"),
      hr(),
      
      h4("Índice kappa"),
      textOutput("kappa"),
      hr(),
      
      h3("Predicciones"),
      tableOutput("predicho_prediccion")
    )
  )
)



server<-function(input,output,session) {
  
  #Inputs usuario
  
  library(dplyr)
  
  procesamiento<-eventReactive(input$ejecutar,{
    req(input$target,input$dominio,input$ubicacion)
    
    data<-read.csv(input$ubicacion$datapath,encoding="UTF-8",na.strings=c("NA",""))
  
    nombre_discretas<-setdiff(colnames(data)[sapply(data,class)=="character"],input$target)
    nombre_continuas<-colnames(data)[sapply(data,is.numeric)]
    
    
    data<-data[,c(nombre_discretas,nombre_continuas,input$target)]
    
    discretas<-which(names(data) %in% nombre_discretas)
    continuas<-which(names(data) %in% nombre_continuas)
    
    dominio_target<-strsplit(input$dominio,",")[[1]]|>trimws()
    
    data$Clase<-as.numeric(factor(data[[input$target]],levels=dominio_target))
    
    longitud_clase<-length(dominio_target)
    dominio_clase<-1:longitud_clase
    
    #Preprocesamiento. Casillas vacías.
    
    nuevas_filas_lista<-list()
    contador<-1
    dominios_columnas<-lapply(names(data),function(col_name){
      na.omit(unique(data[[col_name]]))  
    })
    
    for(i in 1:nrow(data)){
      fila<-data[i,]
      
      if(any(is.na(fila))){
        na_cols<-which(is.na(fila))
        dominios_na_cols<-dominios_columnas[na_cols]
        combinaciones<-expand.grid(dominios_na_cols)
        
        for(j in 1:nrow(combinaciones)){
          nueva_fila<-fila
          for(k in 1:length(na_cols)){
            nueva_fila[[na_cols[k]]]<-combinaciones[j,k]
          }
          nuevas_filas_lista[[contador]]<-nueva_fila
          contador<-contador+1
        }
        
      }else{
        nuevas_filas_lista[[contador]]<-fila
        contador<-contador+1
      }
    }
    
    datos<-do.call(rbind,nuevas_filas_lista)
    
    #División de los datos
    
    library(rsample)
    
    set.seed(10)
    
    split<-initial_split(datos,prop=input$prop_entrenamiento,strata="Clase")
    entrenamiento<-training(split)
    testeo<-testing(split)
    
    #Entrenamiento. Estimación de probabilidades
    
    apriori<-(table(factor(entrenamiento$Clase,levels=dominio_clase))+1)/(nrow(entrenamiento)+longitud_clase)
    
    v_discreta<-lapply(discretas,function(i){
      apply(table(entrenamiento[,i],factor(entrenamiento$Clase,levels=dominio_clase)),2,function(j){
        (j+1)/(sum(j)+nrow(table(entrenamiento[,i])))
      })
    })
    
    v_continua<-array(unlist(
      lapply(dominio_clase,function(i){
        apply(entrenamiento[entrenamiento$Clase==i,continuas],2,function(x){c(mean(x),sd(x))})
      })),dim=c(2,length(continuas), longitud_clase))
    
    #Predicción muestra testeo
    
    predicho_testeo<-apply(testeo,1,function(i){
      densidades_continuas<-if(length(continuas)>0){
        apply(v_continua,3,function(cat){
          prod(dnorm(suppressWarnings(as.numeric(i[continuas])),mean=cat[1,],sd=cat[2,]))})
      }else{
        1
      }
      
      probabilidades_discretas<-if(length(discretas)>0){
        apply(mapply(function(valor,matriz){matriz[valor,]},i[discretas],v_discreta),1,prod)
      }else{
        1
      }
      
      which.max(apriori*densidades_continuas*probabilidades_discretas)
    })
    
    #Métricas de validación
    
    matriz_confusion<-table(
      factor(dominio_target[testeo$Clase],levels=dominio_target),
      factor(dominio_target[predicho_testeo],levels=dominio_target)
    )
    
    exactitud<-sum(diag(matriz_confusion))/nrow(testeo)
    
    recall<-diag(matriz_confusion)/apply(matriz_confusion,1,sum)
    
    precision<-diag(matriz_confusion)/apply(matriz_confusion,2,sum)
    
    f1<-ifelse(precision+recall==0,0,2*precision*recall/(precision+recall))
    
    macro_f1<-mean(f1,na.rm=TRUE)
    
    macro_f1_ponderado<-sum(f1*table(factor(datos$Clase,levels=dominio_clase)),
                            na.rm=TRUE)/nrow(datos)
    
    p_e<-sum(sapply(dominio_clase,function(indice){
      sum(matriz_confusion[indice,])*sum(matriz_confusion[,indice])/nrow(testeo)^2}))
    kappa<-(exactitud-p_e)/(1-p_e)
    
    #Predicción
    
    if(!is.null(input$ubicacion_prediccion)){
      
      prediccion<-read.csv(input$ubicacion_prediccion$datapath,encoding="UTF-8",na.strings=c("NA",""))
      
      prediccion<-prediccion[,c(nombre_discretas,nombre_continuas)]
      
      predicho_prediccion<-apply(prediccion,1,function(i){
        no_na_continuas<-which(!is.na(i[continuas]))
        no_na_discretas<-which(!is.na(i[discretas]))
        
        densidades_continuas<-if(length(no_na_continuas)>0){
          apply(v_continua[,no_na_continuas,,drop=FALSE],3,function(cat){
            prod(dnorm(as.numeric(i[continuas[no_na_continuas]]),mean=cat[1,],sd=cat[2,]))})
        }else{
          1
        }
        
        probabilidades_discretas<-if(length(no_na_discretas)>0){
          apply(mapply(function(valor,matriz){matriz[valor,]},
                       i[discretas[no_na_discretas]],v_discreta[no_na_discretas]),
                1,prod)
        }else{
          1
        }
        
        which.max(apriori*densidades_continuas*probabilidades_discretas)
      })
      
      predicho_prediccion<-dominio_target[predicho_prediccion]
      
    }else{
      predicho_prediccion<-NULL
    }
    
    list(
      matriz_confusion=matriz_confusion,
      exactitud=exactitud,
      recall=recall,
      precision=precision,
      f1=f1,
      macro_f1=macro_f1,
      macro_f1_ponderado=macro_f1_ponderado,
      kappa=kappa,
      predicho_prediccion=predicho_prediccion 
    )
    
  })
    
    
    
  output$matriz_confusion<-renderTable({
    as.data.frame.matrix(procesamiento()$matriz_confusion)
  },rownames=TRUE)
    
  output$exactitud<-renderText({
    round(procesamiento()$exactitud, 3)
  })
  
  
  output$tabla_metricas<-renderTable({
    clases<-strsplit(input$dominio,",")[[1]]|>trimws()
    
    matriz<-rbind(
      Recall=round(procesamiento()$recall,3),
      Precisión=round(procesamiento()$precision,3),
      F1=round(procesamiento()$f1,3)
    )
    
    colnames(matriz)<-clases
    
    as.data.frame(matriz)
  }, rownames = TRUE)
  
  
  
  output$macro_f1<-renderText({
    round(procesamiento()$macro_f1,3)
  })
  
  output$macro_f1_ponderado<-renderText({
    round(procesamiento()$macro_f1_ponderado,3)
  })
  
  output$kappa<-renderText({
    round(procesamiento()$kappa,3)
  })
  
  output$predicho_prediccion<-renderTable({
    data.frame(Observación=seq_along(procesamiento()$predicho_prediccion),
               Predicción=procesamiento()$predicho_prediccion
    )
  })
} 
  


shinyApp(ui=ui,server=server)
  
