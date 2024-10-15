
#.libPaths(c('~/R_libs_go/', .libPaths()))
library(topGO)
library(org.Mm.eg.db)
library(readxl)
library(writexl)

run.traj.enrichment <- function(de.metadata, go.category, output.path) {
  # computes GO enrichment for gene clustered into different aging "trajectories"
  # for oligodendrocytes in the CC/ACO region.
  # de.metadata contains the gene clustering.
  # go.category is the GO hierarchy used (we always use BP).
  # output.path is the path where the output is saved in xlsx format.

  all.traj = unique(unlist(c(de.metadata[,38:ncol(de.metadata)])))
  stopifnot(length(all.traj) == 9)
  
  all.enrichment.results = list()
  for (cur.traj in all.traj) {
    genes = unlist(de.metadata[,'Oligodendrocyte_CC/ACO'])
    names(genes) = de.metadata$gene
    
    genes.fac = as.factor(as.numeric(genes == cur.traj))
    names(genes.fac) = names(genes)
    allGO2genes <- annFUN.org(whichOnto=go.category, feasibleGenes=NULL, mapping="org.Mm.eg.db", ID="symbol")
    GOdata <- new("topGOdata",
      ontology=go.category,
      allGenes=genes.fac,
      annot=annFUN.GO2genes,
      GO2genes=allGO2genes,
      nodeSize=1)
    
    term.to.genes = genesInTerm(GOdata)
    results.fisher <- runTest(GOdata, algorithm="classic", statistic="fisher")
    goEnrichment <- GenTable(GOdata, Fisher=results.fisher, orderBy="Fisher", topNodes=500, numChar=1000)
    stopifnot(tail(goEnrichment, 1)$Fisher > 0.05)
    goEnrichment.fil = goEnrichment[goEnrichment$Fisher < 0.05,]
    genes.per.term = sapply(goEnrichment.fil$GO.ID, function(cur.term) paste(sort(intersect(names(genes)[genes == cur.traj], term.to.genes[[cur.term]])), collapse=' '))
    goEnrichment.fil$genes = genes.per.term
    all.enrichment.results[[cur.traj]] = goEnrichment.fil

  }
  write_xlsx(all.enrichment.results, path=output.path)
}


run.de.enrichment.ci <- function(de.metadata, go.category, output.path) {
  # computes GO enrichment for gene expression that changes with age,
  # using the gene's spearman correlation with age and the correlation's confidence intervals.
  # de.metadata contains the gene correlation data.
  # go.category is the GO hierarchy used (we always use BP).
  # output.path is the path where the output is saved in xlsx format.

  all.ctypes.raw = sapply(strsplit(colnames(de.metadata)[2:73], '_'), function(x) x[1])
  stopifnot(all(all.ctypes.raw[seq(1, length(all.ctypes.raw), by=4)] == all.ctypes.raw[seq(2, length(all.ctypes.raw), by=4)]))
  stopifnot(all(all.ctypes.raw[seq(1, length(all.ctypes.raw), by=4)] == all.ctypes.raw[seq(3, length(all.ctypes.raw), by=4)]))
  stopifnot(all(all.ctypes.raw[seq(1, length(all.ctypes.raw), by=4)] == all.ctypes.raw[seq(4, length(all.ctypes.raw), by=4)]))
  all.ctypes = all.ctypes.raw[seq(1, length(all.ctypes.raw), by=4)]
  all.enrichment.results = list()
  allGO2genes <- annFUN.org(whichOnto=go.category, feasibleGenes=NULL, mapping="org.Mm.eg.db", ID="symbol")
  for (cur.ctype in all.ctypes) {
    for (is.up in c(T, F)) {
      cur.name = sprintf('%s_%s', cur.ctype, c('decreasing', 'increasing')[is.up + 1])
      print(cur.name)
      cur.spearman.cors = unlist(de.metadata[,paste0(cur.ctype, '_Spearman')])
      cur.upper.ci = unlist(de.metadata[,paste0(cur.ctype, '_Upper95CI')])
      cur.lower.ci = unlist(de.metadata[,paste0(cur.ctype, '_Lower95CI')])
      stopifnot(all((cur.spearman.cors > cur.lower.ci) & (cur.spearman.cors < cur.upper.ci)))
      if (is.up) {
        should.select.gene = (cur.spearman.cors > 0.3) & (cur.lower.ci > 0)
      } else {
        should.select.gene = (cur.spearman.cors < -0.3) & (cur.upper.ci < 0)
      }
      genes.fac = as.factor(as.numeric(should.select.gene))
      names(genes.fac) = de.metadata$gene
      cur.genes = de.metadata$gene[should.select.gene]

      GOdata <- new("topGOdata",
        ontology=go.category,
        allGenes=genes.fac,
        annot=annFUN.GO2genes,
        GO2genes=allGO2genes,
        nodeSize=1)

      term.to.genes = genesInTerm(GOdata)
      results.fisher <- runTest(GOdata, algorithm="classic", statistic="fisher")
      total.num.terms = nrow(topGO::termStat(GOdata))
      goEnrichment <- GenTable(GOdata, Fisher=results.fisher, orderBy="Fisher", topNodes=total.num.terms, numChar=1000)
      stopifnot(tail(goEnrichment, 1)$Fisher > 0.05)
      goEnrichment.fil = goEnrichment[goEnrichment$Fisher < 0.05,]
      genes.per.term = sapply(goEnrichment.fil$GO.ID, function(cur.term) paste(sort(intersect(cur.genes, term.to.genes[[cur.term]])), collapse=' '))
      goEnrichment.fil$genes = genes.per.term
      all.enrichment.results[[cur.name]] = goEnrichment.fil
    }
  }
  write_xlsx(all.enrichment.results, path=output.path)
}

run.de.enrichment.ex <- function(de.metadata, intervention.name, go.category, output.path, ignored.ctypes=c()) {
  # computes GO enrichment for gene expression that changes across conditions - both in aging and in exercise.
  # This uses the gene's fold-change and p-value for the change.
  # de.metadata contains the gene fold change data.
  # intervention.name is the name of the tested intervention (we always use Exercise).
  # ignored.ctypes is a vector of cell types to be ignored.
  # go.category is the GO hierarchy used (we always use BP).
  # output.path is the path where the output is saved in xlsx format.
  all.ctypes.raw = sapply(strsplit(colnames(de.metadata)[2:ncol(de.metadata)], '_'), function(x) x[1])
  stopifnot(all(all.ctypes.raw[seq(1, length(all.ctypes.raw), by=4)] == all.ctypes.raw[seq(2, length(all.ctypes.raw), by=4)]))
  stopifnot(all(all.ctypes.raw[seq(1, length(all.ctypes.raw), by=4)] == all.ctypes.raw[seq(3, length(all.ctypes.raw), by=4)]))
  stopifnot(all(all.ctypes.raw[seq(1, length(all.ctypes.raw), by=4)] == all.ctypes.raw[seq(4, length(all.ctypes.raw), by=4)]))
  all.ctypes = all.ctypes.raw[seq(1, length(all.ctypes.raw), by=4)]
  all.enrichment.results = list()
  allGO2genes <- annFUN.org(whichOnto=go.category, feasibleGenes=NULL, mapping="org.Mm.eg.db", ID="symbol")
  for (cur.ctype in all.ctypes) {
    if (cur.ctype %in% ignored.ctypes) {
      next
    }
    for (group.type in c('Aging', intervention.name)) {
      for (is.up in c(T, F)) {
        cur.name = sprintf('%s_%s_%s', cur.ctype, group.type, c('decreasing', 'increasing')[is.up + 1])
        print(cur.name)
        cur.sign = c(-1, 1)[is.up + 1]
	cur.pvals = unlist(de.metadata[,paste0(cur.ctype, '_PValue_', group.type)])
	cur.lfc = sign(log2(unlist(de.metadata[,paste0(cur.ctype, '_FoldChange_', group.type)]))) == cur.sign
	stopifnot(!(any(is.na(cur.lfc) & (cur.pvals != 1))))
        should.select.gene = (cur.pvals < 0.05) & (cur.lfc)
	if (all(!should.select.gene)) {
          all.enrichment.results[[cur.name]] = data.frame()
	  next
	}
        genes.fac = as.factor(as.numeric(should.select.gene))
        names(genes.fac) = de.metadata$gene
        cur.genes = de.metadata$gene[should.select.gene]

        GOdata <- new("topGOdata",
          ontology=go.category,
          allGenes=genes.fac,
          annot=annFUN.GO2genes,
          GO2genes=allGO2genes,
          nodeSize=1)

	total.num.terms = nrow(topGO::termStat(GOdata))
        term.to.genes = genesInTerm(GOdata)
        results.fisher <- runTest(GOdata, algorithm="classic", statistic="fisher")
        goEnrichment <- GenTable(GOdata, Fisher=results.fisher, orderBy="Fisher", topNodes=total.num.terms, numChar=1000)
        stopifnot(tail(goEnrichment, 1)$Fisher > 0.05)
        goEnrichment.fil = goEnrichment[goEnrichment$Fisher < 0.05,]
        genes.per.term = sapply(goEnrichment.fil$GO.ID, function(cur.term) paste(sort(intersect(cur.genes, term.to.genes[[cur.term]])), collapse=' '))
        goEnrichment.fil$genes = genes.per.term
        all.enrichment.results[[cur.name]] = goEnrichment.fil
      }
    }
  }
  write_xlsx(all.enrichment.results, path=output.path)
}

run.clock.enrichment <- function(clock.metadata, go.category, output.path) {
  # computes GO enrichment for the coefficients of genes in aging clocks.
  # clock.metadata contains the genes' coefficients.
  # go.category is the GO hierarchy used (we always use BP).
  # output.path is the path where the output is saved in xlsx format.
  all.ctypes = colnames(clock.metadata)[2:ncol(clock.metadata)]
  all.enrichment.results = list()
  allGO2genes <- annFUN.org(whichOnto=go.category, feasibleGenes=NULL, mapping="org.Mm.eg.db", ID="symbol")
  for (cur.ctype in all.ctypes) {
    for (is.up in c(T, F)) {
      cur.name = sprintf('%s_%s', cur.ctype, c('decreasing', 'increasing')[is.up + 1])
      print(cur.name)
      gene.names = unlist(clock.metadata[,1])
      ctype.coeffs = unlist(clock.metadata[,cur.ctype])
      names(ctype.coeffs) = gene.names
      ctype.pos.coeffs = ctype.coeffs[ctype.coeffs > 0]
      ctype.neg.coeffs = ctype.coeffs[ctype.coeffs < 0]
      sel.pos.genes = names(tail(sort(ctype.pos.coeffs), n=50))
      sel.neg.genes = names(head(sort(ctype.neg.coeffs), n=50))

      if (is.up) {
        genes.fac = as.factor(as.numeric(gene.names %in% sel.pos.genes))
	cur.genes = sel.pos.genes
      } else {
        genes.fac = as.factor(as.numeric(gene.names %in% sel.neg.genes))
	cur.genes = sel.neg.genes
      }
      names(genes.fac) = gene.names

      GOdata <- new("topGOdata",
        ontology=go.category,
        allGenes=genes.fac,
        annot=annFUN.GO2genes,
        GO2genes=allGO2genes,
        nodeSize=1)

      term.to.genes = genesInTerm(GOdata)
      results.fisher <- runTest(GOdata, algorithm="classic", statistic="fisher")
      goEnrichment <- GenTable(GOdata, Fisher=results.fisher, orderBy="Fisher", topNodes=500, numChar=1000)
      stopifnot(tail(goEnrichment, 1)$Fisher > 0.05)
      goEnrichment.fil = goEnrichment[goEnrichment$Fisher < 0.05,]
      genes.per.term = sapply(goEnrichment.fil$GO.ID, function(cur.term) paste(sort(intersect(cur.genes, term.to.genes[[cur.term]])), collapse=' '))
      goEnrichment.fil$genes = genes.per.term
      all.enrichment.results[[cur.name]] = goEnrichment.fil
    }
  }
  write_xlsx(all.enrichment.results, path=output.path)
}


run.enrichments <- function() {
  # main function, that executes all GO enrichment analyses.
  # Input paths to the corresponding results files
  config = list(output_dir='',
                traj_cls_path='',
                dgea_spearman_path='',
		exercise_path='',
		clock_coeffs_path='')

  cls.path = config$traj_cls_path
  output.dir = config$output_dir
  de.metadata = read_excel(cls.path)
  print('running go_oligo_cc_aco_BP')
  run.traj.enrichment(de.metadata, 'BP', file.path(output.dir, 'go_oligo_cc_aco_BP.xlsx'))

  print('running go_de_ci_BP')
  spearman.path = config$dgea_spearman_path
  de.metadata = read.csv(spearman.path)
  run.de.enrichment.ci(de.metadata, 'BP', file.path(output.dir, 'go_de_ci_BP.xlsx'))

  print('running go_de_exercise_BP')
  exercise.path = config$exercise_path
  exer.md = read.csv(exercise.path, stringsAsFactors=F)
  run.de.enrichment.ex(exer.md, 'Exercise', 'BP', file.path(output.dir, 'go_de_exercise_BP.xlsx'))

  print('running go_clock_coeffs_BP')
  clock.file.path = config$clock_coeffs_path
  clock.metadata = read_excel(clock.file.path)
  run.clock.enrichment(clock.metadata, 'BP', file.path(output.dir, 'go_clock_coeffs_BP.xlsx'))
}

