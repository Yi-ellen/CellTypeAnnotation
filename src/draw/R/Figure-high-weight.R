library(VennDiagram)
library(grid)


work_dir <- "../data/Baron_Human/Edge_weight"
setwd(work_dir)

folders <- list.dirs(path = ".", full.names = TRUE, recursive = FALSE)
top_edges_folders <- folders[grepl("_top_edges$", folders)]

fill_colors <- c("#EDADC5", "#CEAAD0", "#9584C1", "#6CBEC3", "#AAD7C8")
line_colors <- c("#EDADC5", "#CEAAD0", "#9584C1", "#6CBEC3", "#AAD7C8")

for (folder in top_edges_folders) {
  cell_type <- gsub("_top_edges", "", basename(folder))
  
  set1 <- read.csv(file.path(folder, "fold_1_edges.csv"))$edges
  set2 <- read.csv(file.path(folder, "fold_2_edges.csv"))$edges
  set3 <- read.csv(file.path(folder, "fold_3_edges.csv"))$edges
  set4 <- read.csv(file.path(folder, "fold_4_edges.csv"))$edges
  set5 <- read.csv(file.path(folder, "fold_5_edges.csv"))$edges
  
  venn_data <- list(Set1 = set1, Set2 = set2, Set3 = set3, Set4 = set4, Set5 = set5)
  
  output_file <- file.path("../../../../result/Figures/Baron_Human/edge", paste0("Baron_Human_", cell_type, "_edge100_venn.svg"))
  
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
    cex = 1,
    
    cat.cex = 1,
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
