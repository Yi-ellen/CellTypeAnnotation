library(VennDiagram)
library(grid)

input_folder <- "../data/AMB/Gene_degree"
output_folder <- "../../../result/Figures/AMB/gene"

tsv_files <- list.files(path = input_folder, pattern = "\\.tsv$", full.names = TRUE)


fill_colors <- c("#EDADC5", "#CEAAD0", "#9584C1", "#6CBEC3", "#AAD7C8")
line_colors <- c("#EDADC5", "#CEAAD0", "#9584C1", "#6CBEC3", "#AAD7C8")

# Figure 7 A-C

for (file_path in tsv_files) {
  file_name <- basename(file_path)  
  cell_type <- sub("_top100_indices\\.tsv$", "", file_name)  
  print(cell_type) 

  data <- read.table(file_path, header = TRUE, sep = "\t")
  

  set1 <- data$Fold_1
  set2 <- data$Fold_2
  set3 <- data$Fold_3
  set4 <- data$Fold_4
  set5 <- data$Fold_5
 
  venn_data <- list(Set1 = set1, Set2 = set2, Set3 = set3, Set4 = set4, Set5 = set5)
  
  output_file <- file.path(output_folder, paste0("AMB_", cell_type, "_g100_venn.svg"))
  
  svg(output_file, width = 4, height = 4)
  
  par(mar = c(3, 3, 3, 3))  
  par(xpd = TRUE)  

  venn.plot <- venn.diagram(
    x = venn_data,
    category.names = c("CV 1", "CV 2", "CV 3", "CV 4", "CV 5"),
    filename = NULL,
    output = TRUE,
    fill = fill_colors,
    lwd = 2,
    col = line_colors,
    alpha = 0.5,
    
    fontfamily = "Times New Roman",
    fontface = "plain",  

    label.col = "black",
    cex = 1.,
    
    cat.cex = 1.,
    cat.fontfamily = "Times New Roman",
    cat.fontface = "plain",   
    cat.col = rgb(1, 1, 1, alpha = 0),
    
    main = cell_type,
    main.fontfamily = "Times New Roman",
    main.fontface = "plain",   
    main.cex = 1.5,            
    main.pos = c(0.5, 0.0)   
  )
  
  plot.new()
  pushViewport(viewport(width = 0.9, height = 0.9))  
  grid.draw(venn.plot)
  upViewport()
  
  par(family = "Times New Roman")
  
  legend(x = grconvertX(0.8, "npc"),  
         y = grconvertY(1.2, "npc"),
         legend = c("CV 1", "CV 2", "CV 3", "CV 4", "CV 5"),
         fill = fill_colors, 
         border = line_colors,
         bty = "n", 
         cex = 0.8,
         inset = c(0.05, 0))

  par(family = "")
  dev.off()
}




# Figure 7D

data <- read.table("../data/AMB/AMB_character_gene_set_indices.tsv", header = TRUE, sep = "\t", fill = TRUE, na.strings ="")

set1 <- unique(na.omit(data$GABAergic)) 
set2 <- unique(na.omit(data$Glutamatergic)) 
set3 <- unique(na.omit(data$NonNeuronal))

venn_data <- list(Set1 = set1, Set2 = set2, Set3 = set3)

fill_colors <- c("#EDADC5", "#9584C1", "#6CBEC3")
line_colors <- c("#EDADC5", "#9584C1", "#6CBEC3")

svg("../../../result/Figures/AMB/gene/AMB_threeTypes_g100_venn.svg", 
    width = 4,   
    height = 4)

par(mar = c(3, 3, 3, 3)) 
par(xpd = TRUE)  


venn.plot <- venn.diagram(
  x = venn_data,
  category.names = c("GABAergic", "Glutamatergic", "Non-Neuronal"),
  filename = NULL,
  output = TRUE,
  fill = fill_colors,
  lwd = 2,
  col = line_colors,
  alpha = 0.5,
  
  fontfamily = "Times New Roman",
  fontface = "plain",  
  
  label.col = "black",
  cex = 1.,
  
  cat.cex = 1.,
  cat.fontfamily = "Times New Roman",
  cat.fontface = "plain",   
  cat.col = rgb(1, 1, 1, alpha = 0),
  
  # main = "Top Degree Genes",
  main.fontfamily = "Times New Roman",
  main.fontface = "plain",   
  main.cex = 1.5,            
  main.pos = c(0.5, 0.0)  
)

plot.new()
pushViewport(viewport(width = 0.72, height = 0.72))  

grid.draw(venn.plot)
upViewport()

par(family = "Times New Roman")
legend(x = grconvertX(0.7, "npc"),  
       y = grconvertY(1.2, "npc"),
       legend = c("GABAergic", "Glutamatergic", "Non-Neuronal"),
       fill = fill_colors, 
       border = line_colors,
       bty = "n", 
       cex = 0.8,
       inset = c(0.05, 0))

par(family = "")
dev.off()





