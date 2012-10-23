package cs224n.assignments;

import cs224n.io.PennTreebankReader;
import cs224n.ling.Tree;
import cs224n.ling.Trees;
import cs224n.parser.EnglishPennTreebankParseEvaluator;
import cs224n.util.*;

import java.util.*;

/**
 * Harness for PCFG Parser project.
 *
 * @author Dan Klein
 */
public class PCFGParserTester {

	// Parser interface ===========================================================

	/**
	 * Parsers are required to map sentences to trees.  How a parser is
	 * constructed and trained is not specified.
	 */
	public static interface Parser {
		public void train(List<Tree<String>> trainTrees);
		public Tree<String> getBestParse(List<String> sentence);
	}


	// PCFGParser =================================================================

	/**
	 * The PCFG Parser you will implement.
	 */
	public static class PCFGParser implements Parser {

		private Grammar grammar;
		private Lexicon lexicon;
		private HashMap<String, Integer> hmStrKey;
		private HashMap<Integer, String> hmIntKey;
		
		
		class BackElement
		{
			public int parent = -1;
			public int child1 = -1;
			public int child2 = -1;
			public int split = -1;
			BackElement(int p, int c1, int c2, int s)
			{
				parent = p;
				child1 = c1;
				child2 = c2;
				split = s;
			}
			BackElement(int p, int c1)
			{
				parent = p;
				child1 = c1;
			}
		}

		@Override
		public void train(List<Tree<String>> trainTrees) {
			// TODO: before you generate your grammar, the training trees
			// need to be binarized so that rules are at most binary
			for(int i = 0; i < trainTrees.size(); i++)
			{
				Tree<String> binaryTree = TreeAnnotations.annotateTree(trainTrees.get(i));
				trainTrees.remove(i);
				trainTrees.add(i, binaryTree);
			}
			lexicon = new Lexicon(trainTrees);
			grammar = new Grammar(trainTrees);
			convertSet2IHm(grammar.tags);
		}

		@Override
		public Tree<String> getBestParse(List<String> sentence) {
			Tree<String> tree = CKY(sentence);
			tree = TreeAnnotations.unAnnotateTree(tree);
			return tree;
		}
		
		private void convertSet2IHm(Set<String> s)
		{
			hmStrKey = new HashMap<String, Integer>();
			hmIntKey = new HashMap<Integer, String>();
			int i= 0;
			for(String tag : s)
			{
				hmStrKey.put(tag, i);
				hmIntKey.put(i, tag);
				i++;
			}
		}
		
		private Tree<String> CKY(List<String> sentence)
		{
			int num_words = sentence.size();
			int num_nonterms = hmStrKey.keySet().size();
			double[][][] score = new double[num_words+1][num_words+1][num_nonterms];
			BackElement[][][] back = new BackElement[num_words+1][num_words+1][num_nonterms];
			String word;
			for(int i = 0; i < num_words; i++)
			{
				word = sentence.get(i);
				for(String A : hmStrKey.keySet())
				{
					int a = hmStrKey.get(A);
					score[i][i+1][a] = lexicon.scoreTagging(word, A);
				}
				
				boolean added = true;
				while(added)
				{
					added = false;
					List<UnaryRule> uRules;
					for(String B : hmStrKey.keySet())
					{
						int b = hmStrKey.get(B);
						double tempScore = score[i][i+1][b];
						if(tempScore > 0)
						{
							uRules = grammar.getUnaryRulesByChild(B);
							for(UnaryRule r : uRules)
							{
								String A = r.getParent();
								double prob = r.getScore()*tempScore;
								int a = hmStrKey.get(A);
								if(prob > score[i][i+1][a])
								{
									score[i][i+1][a] = prob;
									back[i][i+1][a] = new BackElement(a,b);
									added = true;
								}
							}
						}
					}
				}
			}
			
			for(int span = 2; span <= num_words; span++)
			{
				for(int begin = 0; begin <= num_words-span; begin++)
				{
					int end = begin + span;
					for(int split = begin + 1; split <= end - 1; split ++)
					{
						List<BinaryRule> bRules;
						for(String B : hmStrKey.keySet())
						{
							int b = hmStrKey.get(B);
							double tempScoreB = score[begin][split][b];
							if(tempScoreB == 0)
								continue;
							bRules = grammar.getBinaryRulesByLeftChild(B);
							for(BinaryRule r : bRules)
							{
								String A = r.getParent();
								String C = r.getRightChild();
								int a = hmStrKey.get(A);
								int c = hmStrKey.get(C);
								
								double tempScoreC = score[split][end][c];
								if(tempScoreC == 0)
									continue;
								double prob = tempScoreB*tempScoreC*r.getScore();
								
								double temp_val = score[begin][end][a];
								if(new Double(prob).doubleValue() > new Double(temp_val).doubleValue())
								{
									score[begin][end][a] = prob; 
									back[begin][end][a] = new BackElement(a, b, c, split);
								}
							}
						}
					}
					boolean added = true;
					while(added)
					{
						added = false;
						List<UnaryRule> uRules;
						for(String B : hmStrKey.keySet())
						{
							int b = hmStrKey.get(B);
							double tempScore = score[begin][end][b];
							if(tempScore == 0)
								continue;
							uRules = grammar.getUnaryRulesByChild(B);
							for(UnaryRule r : uRules)
							{
								String A = r.getParent();
								int a = hmStrKey.get(A);
								double prob = r.getScore()*tempScore;
								double temp_val = score[begin][end][a];
								if(new Double(prob).doubleValue() > new Double(temp_val).doubleValue())
								{
									score[begin][end][a] = prob;
									back[begin][end][a] = new BackElement(a,b);
									added = true;
								}
							}
						}
					}
				}
			}
			//find the highest prob for the root
			int index = hmStrKey.get("ROOT");
			return BuildTree(0, num_words, back, index, sentence);
		}
		
		private Tree<String> BuildTree(int lIndex, int rIndex, BackElement[][][] back, int bestIndex, List<String> sentence)
		{
			String parentLabel = hmIntKey.get(bestIndex);
			Tree<String> tree = new Tree<String>(parentLabel);
			List<Tree<String>> children = new ArrayList<Tree<String>>();
			BackElement e = back[lIndex][rIndex][bestIndex];
			if(lIndex == rIndex - 1 && e == null)
			{
				Tree<String> child = new Tree<String>(sentence.get(lIndex));
				children.add(child);
				tree.setChildren(children);
				return tree;
			}
			
			
			
			int child1 = e.child1;
			int child2 = 0;
			if(e.split != -1)
				child2 = e.child2;
				
			
			
			if(e.split == -1)
			{
				children.add(BuildTree(lIndex, rIndex, back, child1, sentence));
			}
			else
			{
				children.add(BuildTree(lIndex, e.split, back, child1, sentence));
				children.add(BuildTree(e.split, rIndex, back, child2, sentence));
			}
			tree.setChildren(children);
			return tree;
		}
	}
	

	// BaselineParser =============================================================

	/**
	 * Baseline parser (though not a baseline I've ever seen before).  Tags the
	 * sentence using the baseline tagging method, then either retrieves a known
	 * parse of that tag sequence, or builds a right-branching parse for unknown
	 * tag sequences.
	 */
	public static class BaselineParser implements Parser {

		CounterMap<List<String>,Tree<String>> knownParses;
		CounterMap<Integer,String> spanToCategories;
		Lexicon lexicon;

		@Override
		public void train(List<Tree<String>> trainTrees) {
			lexicon = new Lexicon(trainTrees);
			knownParses = new CounterMap<List<String>, Tree<String>>();
			spanToCategories = new CounterMap<Integer, String>();
			for (Tree<String> trainTree : trainTrees) {
				List<String> tags = trainTree.getPreTerminalYield();
				knownParses.incrementCount(tags, trainTree, 1.0);
				tallySpans(trainTree, 0);
			}
		}

		@Override
		public Tree<String> getBestParse(List<String> sentence) {
			List<String> tags = getBaselineTagging(sentence);
			if (knownParses.keySet().contains(tags)) {
				return getBestKnownParse(tags, sentence);
			}
			return buildRightBranchParse(sentence, tags);
		}

		/* Builds a tree that branches to the right.  For pre-terminals it
		 * uses the most common tag for the word in the training corpus.
		 * For all other non-terminals it uses the tag that is most common
		 * in training corpus of tree of the same size span as the tree
		 * that is being labeled. */
		private Tree<String> buildRightBranchParse(List<String> words, List<String> tags) {
			int currentPosition = words.size() - 1;
			Tree<String> rightBranchTree = buildTagTree(words, tags, currentPosition);
			while (currentPosition > 0) {
				currentPosition--;
				rightBranchTree = merge(buildTagTree(words, tags, currentPosition),
						rightBranchTree);
			}
			rightBranchTree = addRoot(rightBranchTree);
			return rightBranchTree;
		}

		private Tree<String> merge(Tree<String> leftTree, Tree<String> rightTree) {
			int span = leftTree.getYield().size() + rightTree.getYield().size();
			String mostFrequentLabel = spanToCategories.getCounter(span).argMax();
			List<Tree<String>> children = new ArrayList<Tree<String>>();
			children.add(leftTree);
			children.add(rightTree);
			return new Tree<String>(mostFrequentLabel, children);
		}

		private Tree<String> addRoot(Tree<String> tree) {
			return new Tree<String>("ROOT", Collections.singletonList(tree));
		}

		private Tree<String> buildTagTree(List<String> words,
				List<String> tags,
				int currentPosition) {
			Tree<String> leafTree = new Tree<String>(words.get(currentPosition));
			Tree<String> tagTree = new Tree<String>(tags.get(currentPosition), 
					Collections.singletonList(leafTree));
			return tagTree;
		}

		private Tree<String> getBestKnownParse(List<String> tags, List<String> sentence) {
			Tree<String> parse = knownParses.getCounter(tags).argMax().deepCopy();
			parse.setWords(sentence);
			return parse;
		}

		private List<String> getBaselineTagging(List<String> sentence) {
			List<String> tags = new ArrayList<String>();
			for (String word : sentence) {
				String tag = getBestTag(word);
				tags.add(tag);
			}
			return tags;
		}

		private String getBestTag(String word) {
			double bestScore = Double.NEGATIVE_INFINITY;
			String bestTag = null;
			for (String tag : lexicon.getAllTags()) {
				double score = lexicon.scoreTagging(word, tag);
				if (bestTag == null || score > bestScore) {
					bestScore = score;
					bestTag = tag;
				}
			}
			return bestTag;
		}

		private int tallySpans(Tree<String> tree, int start) {
			if (tree.isLeaf() || tree.isPreTerminal()) 
				return 1;
			int end = start;
			for (Tree<String> child : tree.getChildren()) {
				int childSpan = tallySpans(child, end);
				end += childSpan;
			}
			String category = tree.getLabel();
			if (! category.equals("ROOT"))
				spanToCategories.incrementCount(end - start, category, 1.0);
			return end - start;
		}

	}


	// TreeAnnotations ============================================================

	/**
	 * Class which contains code for annotating and binarizing trees for
	 * the parser's use, and debinarizing and unannotating them for
	 * scoring.
	 */
	public static class TreeAnnotations {

		private static void Markov1(Tree<String> Input_Tree,String Parent){
			String cur_label = Input_Tree.getLabel();
			if(Input_Tree.isLeaf()) return;
			
			Input_Tree.setLabel(cur_label + (Parent==null?"":"^"+Parent));
			
			for( Tree<String> Child: Input_Tree.getChildren()){
				Markov1(Child, cur_label);
			}			
		}
		
		private static void Markov2(Tree<String> Input_Tree,String Parent, String GrandParent){
			String cur_label = Input_Tree.getLabel();
			if(Input_Tree.isLeaf()) return;
			
			Input_Tree.setLabel(cur_label + (Parent==null?"":"^"+Parent) + (GrandParent==null?"":"^"+GrandParent));
			
			for( Tree<String> Child: Input_Tree.getChildren()){
				Markov2(Child, cur_label, Parent);
			}			
		}
		
		private static void Markov3(Tree<String> Input_Tree,String Parent, String GrandParent, String GranGrandParent){
			String cur_label = Input_Tree.getLabel();
			if(Input_Tree.isLeaf()) return;
			
			Input_Tree.setLabel(cur_label + (Parent==null?"":"^"+Parent) + (GrandParent==null?"":"^"+GrandParent)
					+ (GranGrandParent==null?"":"^"+GranGrandParent));
			
			for( Tree<String> Child: Input_Tree.getChildren()){
				Markov3(Child, cur_label, Parent, GrandParent);
			}			
		}
		
		
		private static void HorizonOnly_Markov(Tree<String> Input_Tree, Tree<String> Parent){
			String cur_label = Input_Tree.getLabel();
			if(Input_Tree.isLeaf()) return;
			
			//horizontal
			if(Parent != null){
				for( Tree<String> Sibling: Parent.getChildren()){
					if(Sibling == null) continue;
					if(Sibling.isLeaf()) continue;
					cur_label+="_" + Sibling.getLabel();		
				}
				Input_Tree.setLabel(cur_label);		
			}
			
						
			for( Tree<String> Child: Input_Tree.getChildren()){
				HorizonOnly_Markov(Child, Input_Tree);
			}			
		}
		
		private static void Horizon_Markov2V(Tree<String> Input_Tree, Tree<String> Parent,String parentTag, String GrandparentTag){
			String cur_label = Input_Tree.getLabel();
			String backup = cur_label;
			if(Input_Tree.isLeaf()) return;
			
			//horizontal
			if(Parent != null){
				for( Tree<String> Sibling: Parent.getChildren()){
					if(Sibling == null) continue;
					if(Sibling.isLeaf()) continue;
					cur_label+= "_" + Sibling.getLabel();
				}				
			}
			
			//2nd order vertical
			Input_Tree.setLabel(cur_label + (Parent==null?"":"^"+Parent.getLabel()) + (GrandparentTag==null?"":"^"+GrandparentTag));
			
			for( Tree<String> Child: Input_Tree.getChildren()){
				Horizon_Markov2V(Child, Input_Tree, backup, parentTag);
			}			
		}
		
		public static Tree<String> annotateTree(Tree<String> unAnnotatedTree) {

			// Currently, the only annotation done is a lossless binarization

			// TODO: change the annotation from a lossless binarization to a
			// finite-order markov process (try at least 1st and 2nd order)
			
			//HorizonOnly_Markov(unAnnotatedTree, null);
			//Markov2(unAnnotatedTree, null, null);
			Horizon_Markov2V(unAnnotatedTree, null, null, null);
			
			// TODO : mark nodes with the label of their parent nodes, giving a second
			// order vertical markov process
			return binarizeTree(unAnnotatedTree);

		}

		private static Tree<String> binarizeTree(Tree<String> tree) {
			String label = tree.getLabel();
			if (tree.isLeaf())
				return new Tree<String>(label);
			if (tree.getChildren().size() == 1) {
				return new Tree<String>
				(label, 
						Collections.singletonList(binarizeTree(tree.getChildren().get(0))));
			}
			// otherwise, it's a binary-or-more local tree, 
			// so decompose it into a sequence of binary and unary trees.
			String intermediateLabel = "@"+label+"->";
			Tree<String> intermediateTree =
					binarizeTreeHelper(tree, 0, intermediateLabel);
			return new Tree<String>(label, intermediateTree.getChildren());
		}

		private static Tree<String> binarizeTreeHelper(Tree<String> tree,
				int numChildrenGenerated, 
				String intermediateLabel) {
			Tree<String> leftTree = tree.getChildren().get(numChildrenGenerated);
			List<Tree<String>> children = new ArrayList<Tree<String>>();
			children.add(binarizeTree(leftTree));
			if (numChildrenGenerated < tree.getChildren().size() - 1) {
				Tree<String> rightTree = 
						binarizeTreeHelper(tree, numChildrenGenerated + 1, 
								intermediateLabel + "_" + leftTree.getLabel());
				children.add(rightTree);
			}
			return new Tree<String>(intermediateLabel, children);
		} 

		public static Tree<String> unAnnotateTree(Tree<String> annotatedTree) {

			// Remove intermediate nodes (labels beginning with "@"
			// Remove all material on node labels which follow their base symbol 
			// (cuts at the leftmost -, ^, or : character)
			// Examples: a node with label @NP->DT_JJ will be spliced out, 
			// and a node with label NP^S will be reduced to NP

			Tree<String> debinarizedTree =
					Trees.spliceNodes(annotatedTree, new Filter<String>() {
						@Override
						public boolean accept(String s) {
							return s.startsWith("@");
						}
					});
			Tree<String> unAnnotatedTree = 
					(new Trees.FunctionNodeStripper()).transformTree(debinarizedTree);
			return unAnnotatedTree;
		}
	}


	// Lexicon ====================================================================

	/**
	 * Simple default implementation of a lexicon, which scores word,
	 * tag pairs with a smoothed estimate of P(tag|word)/P(tag).
	 */
	public static class Lexicon {

		CounterMap<String,String> wordToTagCounters = new CounterMap<String, String>();
		double totalTokens = 0.0;
		double totalWordTypes = 0.0;
		Counter<String> tagCounter = new Counter<String>();
		Counter<String> wordCounter = new Counter<String>();
		Counter<String> typeTagCounter = new Counter<String>();

		public Set<String> getAllTags() {
			return tagCounter.keySet();
		}

		public boolean isKnown(String word) {
			return wordCounter.keySet().contains(word);
		}

		/* Returns a smoothed estimate of P(word|tag) */
		public double scoreTagging(String word, String tag) {
			double p_tag = tagCounter.getCount(tag) / totalTokens;
			double c_word = wordCounter.getCount(word);
			double c_tag_and_word = wordToTagCounters.getCount(word, tag);
			if (c_word < 10) { // rare or unknown
				c_word += 1.0;
				c_tag_and_word += typeTagCounter.getCount(tag) / totalWordTypes;
			}
			double p_word = (1.0 + c_word) / (totalTokens + totalWordTypes);
			double p_tag_given_word = c_tag_and_word / c_word;
			return ( p_tag == 0 || p_word == 0)?0:(p_tag_given_word / p_tag * p_word);
		}

		/* Builds a lexicon from the observed tags in a list of training trees. */
		public Lexicon(List<Tree<String>> trainTrees) {
			for (Tree<String> trainTree : trainTrees) {
				List<String> words = trainTree.getYield();
				List<String> tags = trainTree.getPreTerminalYield();
				for (int position = 0; position < words.size(); position++) {
					String word = words.get(position);
					String tag = tags.get(position);
					tallyTagging(word, tag);
				}
			}
		}

		private void tallyTagging(String word, String tag) {
			if (! isKnown(word)) {
				totalWordTypes += 1.0;
				typeTagCounter.incrementCount(tag, 1.0);
			}
			totalTokens += 1.0;
			tagCounter.incrementCount(tag, 1.0);
			wordCounter.incrementCount(word, 1.0);
			wordToTagCounters.incrementCount(word, tag, 1.0);
		}
	}


	// Grammar ====================================================================

	/**
	 * Simple implementation of a PCFG grammar, offering the ability to
	 * look up rules by their child symbols.  Rule probability estimates
	 * are just relative frequency estimates off of training trees.
	 */
	public static class Grammar {

		Map<String, List<BinaryRule>> binaryRulesByLeftChild = 
				new HashMap<String, List<BinaryRule>>();
		Map<String, List<BinaryRule>> binaryRulesByRightChild = 
				new HashMap<String, List<BinaryRule>>();
		Map<String, List<UnaryRule>> unaryRulesByChild = 
				new HashMap<String, List<UnaryRule>>();
		Set<String> tags = new HashSet<String>();

		/* Rules in grammar are indexed by child for easy access when
		 * doing bottom up parsing. */
		public List<BinaryRule> getBinaryRulesByLeftChild(String leftChild) {
			return CollectionUtils.getValueList(binaryRulesByLeftChild, leftChild);
		}

		public List<BinaryRule> getBinaryRulesByRightChild(String rightChild) {
			return CollectionUtils.getValueList(binaryRulesByRightChild, rightChild);
		}

		public List<UnaryRule> getUnaryRulesByChild(String child) {
			return CollectionUtils.getValueList(unaryRulesByChild, child);
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			List<String> ruleStrings = new ArrayList<String>();
			for (String leftChild : binaryRulesByLeftChild.keySet()) {
				for (BinaryRule binaryRule : getBinaryRulesByLeftChild(leftChild)) {
					ruleStrings.add(binaryRule.toString());
				}
			}
			for (String child : unaryRulesByChild.keySet()) {
				for (UnaryRule unaryRule : getUnaryRulesByChild(child)) {
					ruleStrings.add(unaryRule.toString());
				}
			}
			for (String ruleString : CollectionUtils.sort(ruleStrings)) {
				sb.append(ruleString);
				sb.append("\n");
			}
			return sb.toString();
		}

		private void addBinary(BinaryRule binaryRule) {
			CollectionUtils.addToValueList(binaryRulesByLeftChild, 
					binaryRule.getLeftChild(), binaryRule);
			CollectionUtils.addToValueList(binaryRulesByRightChild, 
					binaryRule.getRightChild(), binaryRule);
		}

		private void addUnary(UnaryRule unaryRule) {
			CollectionUtils.addToValueList(unaryRulesByChild, 
					unaryRule.getChild(), unaryRule);
		}

		/* A builds PCFG using the observed counts of binary and unary
		 * productions in the training trees to estimate the probabilities
		 * for those rules.  */ 
		public Grammar(List<Tree<String>> trainTrees) {
			Counter<UnaryRule> unaryRuleCounter = new Counter<UnaryRule>();
			Counter<BinaryRule> binaryRuleCounter = new Counter<BinaryRule>();
			Counter<String> symbolCounter = new Counter<String>();
			for (Tree<String> trainTree : trainTrees) {
				tallyTree(trainTree, symbolCounter, unaryRuleCounter, binaryRuleCounter);
			}
			for (UnaryRule unaryRule : unaryRuleCounter.keySet()) {
				double unaryProbability = 
						unaryRuleCounter.getCount(unaryRule) / 
						symbolCounter.getCount(unaryRule.getParent());
				unaryRule.setScore(unaryProbability);
				addUnary(unaryRule);
			}
			for (BinaryRule binaryRule : binaryRuleCounter.keySet()) {
				double binaryProbability = 
						binaryRuleCounter.getCount(binaryRule) / 
						symbolCounter.getCount(binaryRule.getParent());
				binaryRule.setScore(binaryProbability);
				addBinary(binaryRule);
			}
		}

		private void tallyTree(Tree<String> tree, Counter<String> symbolCounter,
				Counter<UnaryRule> unaryRuleCounter, 
				Counter<BinaryRule> binaryRuleCounter) {
			if (tree.isLeaf()) return;
			if (tree.isPreTerminal()) return;
			
			if (tree.getChildren().size() == 1) {
				UnaryRule unaryRule = makeUnaryRule(tree);
				symbolCounter.incrementCount(tree.getLabel(), 1.0);
				unaryRuleCounter.incrementCount(unaryRule, 1.0);
				tags.add(tree.getLabel());
				tags.add(tree.getChildren().get(0).getLabel());
			}
			if (tree.getChildren().size() == 2) {
				BinaryRule binaryRule = makeBinaryRule(tree);
				symbolCounter.incrementCount(tree.getLabel(), 1.0);
				binaryRuleCounter.incrementCount(binaryRule, 1.0);
				tags.add(tree.getLabel());
				tags.add(tree.getChildren().get(0).getLabel());
				tags.add(tree.getChildren().get(1).getLabel());
			}
			if (tree.getChildren().size() < 1 || tree.getChildren().size() > 2) {
				throw new RuntimeException("Attempted to construct a Grammar with an illegal tree: "+tree +" "+tree.getChildren().size());
			}
			for (Tree<String> child : tree.getChildren()) {
				tallyTree(child, symbolCounter, unaryRuleCounter,  binaryRuleCounter);
			}
		}

		private UnaryRule makeUnaryRule(Tree<String> tree) {
			return new UnaryRule(tree.getLabel(), tree.getChildren().get(0).getLabel());
		}

		private BinaryRule makeBinaryRule(Tree<String> tree) {
			return new BinaryRule(tree.getLabel(), tree.getChildren().get(0).getLabel(), 
					tree.getChildren().get(1).getLabel());
		}
	}


	// BinaryRule =================================================================

	/* A binary grammar rule with score representing its probability. */
	public static class BinaryRule {

		String parent;
		String leftChild;
		String rightChild;
		double score;

		public String getParent() {
			return parent;
		}

		public String getLeftChild() {
			return leftChild;
		}

		public String getRightChild() {
			return rightChild;
		}

		public double getScore() {
			return score;
		}

		public void setScore(double score) {
			this.score = score;
		}

		@Override
		public boolean equals(Object o) {
			if (this == o) return true;
			if (!(o instanceof BinaryRule)) return false;

			final BinaryRule binaryRule = (BinaryRule) o;

			if (leftChild != null ? !leftChild.equals(binaryRule.leftChild) : binaryRule.leftChild != null) 
				return false;
			if (parent != null ? !parent.equals(binaryRule.parent) : binaryRule.parent != null) 
				return false;
			if (rightChild != null ? !rightChild.equals(binaryRule.rightChild) : binaryRule.rightChild != null) 
				return false;

			return true;
		}

		@Override
		public int hashCode() {
			int result;
			result = (parent != null ? parent.hashCode() : 0);
			result = 29 * result + (leftChild != null ? leftChild.hashCode() : 0);
			result = 29 * result + (rightChild != null ? rightChild.hashCode() : 0);
			return result;
		}

		@Override
		public String toString() {
			return parent + " -> " + leftChild + " " + rightChild + " %% "+score;
		}

		public BinaryRule(String parent, String leftChild, String rightChild) {
			this.parent = parent;
			this.leftChild = leftChild;
			this.rightChild = rightChild;
		}
	}


	// UnaryRule ==================================================================

	/** A unary grammar rule with score representing its probability. */
	public static class UnaryRule {

		String parent;
		String child;
		double score;

		public String getParent() {
			return parent;
		}

		public String getChild() {
			return child;
		}

		public double getScore() {
			return score;
		}

		public void setScore(double score) {
			this.score = score;
		}

		@Override
		public boolean equals(Object o) {
			if (this == o) return true;
			if (!(o instanceof UnaryRule)) return false;

			final UnaryRule unaryRule = (UnaryRule) o;

			if (child != null ? !child.equals(unaryRule.child) : unaryRule.child != null) return false;
			if (parent != null ? !parent.equals(unaryRule.parent) : unaryRule.parent != null) return false;

			return true;
		}

		@Override
		public int hashCode() {
			int result;
			result = (parent != null ? parent.hashCode() : 0);
			result = 29 * result + (child != null ? child.hashCode() : 0);
			return result;
		}

		@Override
		public String toString() {
			return parent + " -> " + child + " %% "+score;
		}

		public UnaryRule(String parent, String child) {
			this.parent = parent;
			this.child = child;
		}
	}


	// PCFGParserTester ===========================================================

	// Longest sentence length that will be tested on.
	private static int MAX_LENGTH = 20;

	private static void testParser(Parser parser, List<Tree<String>> testTrees) {
		EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> eval = 
				new EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String>
		(Collections.singleton("ROOT"), 
				new HashSet<String>(Arrays.asList(new String[] {"''", "``", ".", ":", ","})));
		for (Tree<String> testTree : testTrees) {
			List<String> testSentence = testTree.getYield();
			if (testSentence.size() > MAX_LENGTH)
				continue;
			Tree<String> guessedTree = parser.getBestParse(testSentence);
			System.out.println("Guess:\n"+Trees.PennTreeRenderer.render(guessedTree));
			System.out.println("Gold:\n"+Trees.PennTreeRenderer.render(testTree));
			eval.evaluate(guessedTree, testTree);
		}
		eval.display(true);
	}

	private static List<Tree<String>> readTrees(String basePath, int low,
			int high) {
		Collection<Tree<String>> trees = PennTreebankReader.readTrees(basePath,
				low, high);
		// normalize trees
		Trees.TreeTransformer<String> treeTransformer = new Trees.StandardTreeNormalizer();
		List<Tree<String>> normalizedTreeList = new ArrayList<Tree<String>>();
		for (Tree<String> tree : trees) {
			Tree<String> normalizedTree = treeTransformer.transformTree(tree);
			// System.out.println(Trees.PennTreeRenderer.render(normalizedTree));
			normalizedTreeList.add(normalizedTree);
		}
		return normalizedTreeList;
	}

	public static void main(String[] args) {

		// set up default options ..............................................
		Map<String, String> options = new HashMap<String, String>();
		options.put("-path",      "/afs/ir/class/cs224n/pa2/data/");
		options.put("-data",      "miniTest");
		//options.put("-data",      "treebank");
		//options.put("-parser",    "cs224n.assignments.PCFGParserTester$BaselineParser");
		options.put("-parser",    "cs224n.assignments.PCFGParserTester$PCFGParser");
		options.put("-maxLength", "20");

		// let command-line options supersede defaults .........................
		options.putAll(CommandLineUtils.simpleCommandLineParser(args));
		System.out.println("PCFGParserTester options:");
		for (Map.Entry<String, String> entry: options.entrySet()) {
			System.out.printf("  %-12s: %s%n", entry.getKey(), entry.getValue());
		}
		System.out.println();

		MAX_LENGTH = Integer.parseInt(options.get("-maxLength"));

		Parser parser;
		try {
			Class parserClass = Class.forName(options.get("-parser"));
			parser = (Parser) parserClass.newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		System.out.println("Using parser: " + parser);

		String basePath = options.get("-path");
		String dataSet = options.get("-data");
		if (!basePath.endsWith("/"))
			basePath += "/";
		//basePath += dataSet;
		System.out.println("Data will be loaded from: " + basePath + "\n");

		List<Tree<String>> trainTrees = new ArrayList<Tree<String>>(),
				validationTrees = new ArrayList<Tree<String>>(),
				testTrees = new ArrayList<Tree<String>>();

		if (!basePath.endsWith("/"))
			basePath += "/";
		basePath += dataSet;
		if (dataSet.equals("miniTest")) {
			System.out.print("Loading training trees...");
			trainTrees = readTrees(basePath, 1, 3);
			System.out.println("done.");
			System.out.print("Loading test trees...");
			testTrees = readTrees(basePath, 4, 4);
			System.out.println("done.");
		}
		else if (dataSet.equals("treebank")) {
			System.out.print("Loading training trees...");
			trainTrees = readTrees(basePath, 200, 2199);
			System.out.println("done.");
			System.out.print("Loading validation trees...");
			validationTrees = readTrees(basePath, 2200, 2299);
			System.out.println("done.");
			System.out.print("Loading test trees...");
			testTrees = readTrees(basePath, 2300, 2319);
			System.out.println("done.");
		}
		else {
			throw new RuntimeException("Bad data set mode: "+ dataSet+", use miniTest, or treebank."); 
		}
		parser.train(trainTrees);
		testParser(parser, testTrees);
	}
}
