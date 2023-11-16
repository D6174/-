package com.classify;

import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.nd4j.common.primitives.Pair;

import java.util.*;

/**
 * @author lh
 * @version 1.0
 */
public class MySentenceProvider implements LabeledSentenceProvider {
    private final int totalCount;
    private final List<String> allLabels;
    private List<String> examples = new ArrayList<>();
    private int cursor;


    public MySentenceProvider(List<String> examples) {
        totalCount = examples.size();
        allLabels = new ArrayList<>();

        Set<String> labelSet = new HashSet<>();
        for (String example : examples) {
            int index = example.indexOf(" ");
            if (index == -1) continue;
            this.examples.add(example);
            String label = example.substring(0, index);
            if (labelSet.contains(label)) {
                continue;
            }
            labelSet.add(label);
            allLabels.add(label);
        }
        Collections.sort(allLabels);
    }

    @Override
    public boolean hasNext() {
        return this.cursor < this.totalCount - 1;
    }

    @Override
    public Pair<String, String> nextSentence() {
        String example = examples.get(this.cursor);
        int index = example.indexOf(" ");
        String label = example.substring(0, index);
        String sentence = example.substring(index + 1, example.length());
        this.cursor ++;
        return new Pair(sentence, label);
    }

    @Override
    public void reset() {
        this.cursor = 0;
    }

    @Override
    public int totalNumSentences() {
        return this.totalCount;
    }

    @Override
    public List<String> allLabels() {
        return this.allLabels;
    }

    @Override
    public int numLabelClasses() {
        return this.allLabels.size();
    }
}
