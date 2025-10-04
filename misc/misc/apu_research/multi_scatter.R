library(ggplot2)
library(readr)
setwd("C:\\GoogleDrive\\GitHub\\deep_learning\\misc\\misc\\apu_research")
df = read.csv('summary_stats_sovi_svi_LI_dynamic0711.csv')
df$LI='Low Income'
df2 = read.csv('summary_stats_sovi_svi_NLI_dynamic0711.csv')
df2$LI='not-Low Income'
df=rbind(df,df2)

x=df[df$Source=='SOVI',c('cliff_d','LI','Cregion')]
y=df[df$Source=='SVI',c('cliff_d','LI','Cregion')]


library(ggplot2)
library(patchwork)
library(cowplot)
library(dplyr)

P <- list()
print(unique(df$Source))
for (zz in unique(df$Source)) {
  if (zz != 'SOVI' & zz != 'SVI') {
    
    z <- df[df$Source == zz, c('cliff_d', 'LI', 'Cregion')]
    df1 <- merge(x, y, by = c('Cregion', 'LI'))
    df1 <- merge(df1, z, by = c('Cregion', 'LI'))
    names(df1) <- c('Cregion', 'Li', 'SOVI', 'SVI', 'J40')
    
    # Create each plot without individual legends
    p <- ggplot(df1, aes(x = SOVI, y = SVI)) +
      annotate("rect", xmin = 0, xmax = 1, ymin = 0, ymax = 1,
               color = "darkred", fill = NA, linetype = "dashed", size = 1.2) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray") +
      geom_point(aes(color = J40, shape = Li), size = 4, alpha = 0.8) +
      # scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, limits = c(-1, 1),
      #                       name = "Cliff's δ (J40)\n−1: Higher in independent\n+1: Higher in compound") +
      scale_color_gradient2(
        low = "blue", mid = "white", high = "red",
        midpoint = 0, limits = c(-1, 1),
        name = "Cliff's δ (J40)",
        breaks = c(-1, 0, 1),
        labels = c("Higher in \n independent", "Equal", "Higher in \n compound")
      )+
      scale_shape_manual(values = c(16, 17)) +
      xlim(c(-1, 1)) + ylim(c(-1, 1)) +
      scale_x_continuous(limits = c(-1, 1), expand = expansion(mult = 0.1)) +
      scale_y_continuous(limits = c(-1, 1), expand = expansion(mult = 0.1))+
      geom_text(aes(label = Cregion), hjust = -0.2, vjust = -0.2, size = 2.5) +
      #theme_minimal() +
      theme(panel.background = element_rect(fill = "white", colour = "grey50"))+
      labs(
        title = zz,
        x = "Cliff's δ (SOVI)", y = "Cliff's δ (SVI)", shape = "Income"
      ) +
      theme(legend.position = "none",
            plot.title = element_text(size = 9),  # 🔹 Reduce plot title size
            axis.title = element_text(size = 8)                  # Optional: axis labels
            #axis.text = element_text(size = 8)                    # Optional: axis tick labels
            
      )
    
    P[[zz]] <- p
  }
}

# Extract legend from one of the plots (e.g., first plot in P)
first_plot <- P[[1]]
legend <- get_legend(
  P[[1]] +
    theme(
      legend.position = "bottom",
      legend.box = "vertical",
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8)
    ) +
    guides(
      color = guide_colorbar(
        title.position = "top",
        direction = "horizontal",
        barwidth = unit(4, "cm"),
        barheight = unit(0.4, "cm")
      ),
      shape = guide_legend(
        title = "Income",
        title.position = "top",
        ncol = 1,
        override.aes = list(size = 4)
      )
    )
) 


# Add the legend to the 16th slot (bottom-right of 4x4 grid)
legend_plot <- wrap_elements(legend)
P[["__legend__"]] <- legend_plot
final_plot <- wrap_plots(P, ncol = 4)
final_plot
