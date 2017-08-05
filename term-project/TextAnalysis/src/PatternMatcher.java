package utexas.parallel;

import java.util.ArrayList;
import java.util.OptionalInt;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class PatternMatcher {
    char[] pattern;
    int[] witness;

    public PatternMatcher(String pattern) {
        this(pattern, generateWitness(pattern.toCharArray()));
    }

    public PatternMatcher(String pattern, int[] witness) {
        this.pattern = pattern.toCharArray();
        this.witness = witness;
    }

    public static String getPeriod(String input) {
        return input.substring(0, getPeriodLength(input.toCharArray()));
    }

    public static int getPeriodLength(char[] input) {
        start:
        for (int i = 0; i < input.length; ) {
            for (int j = i; j < input.length; j++) {
                if (input[j % (i + 1)] != input[j]) {
                    i = j;
                    continue start;
                }
            }
            return i + 1;
        }
        assert false; // Shouldn't happen
        return -1;
    }

    static int[] generateWitness(char[] pattern) {
        int witnessLength = Math.max(2, Math.min((int) Math.ceil(pattern.length / 2.0), getPeriodLength(pattern)));
        int[] witness = new int[witnessLength];

        for (int offset = 0; offset < witness.length; offset++) {
            // Get 0 based index of the first differing character of
            // the shifted pattern
            for (int i = 0; i < witness.length; i++) {
                if (pattern[i + offset] != pattern[i]) {
                    witness[offset] = i;
                    break;
                }

                // shifted pattern matched exactly for every element
                if (i == witness.length - 1) {
                    witness[offset] = -1;
                }
            }
        }

        return witness;
    }

    public int duel(char[] text, int i, int j) {
        int k = witness[j - i];

        if (j + k >= text.length || text[j + k] != pattern[k])
            return i;
        else
            return j;
    }

    public int duel(String text, int i, int j) {
        return duel(text.toCharArray(), i, j);
    }

    public ArrayList<Integer> getMatchedPositions(char[] text) {
        ArrayList<Integer> duelWinners = getDuelWinners(text);
        ArrayList<Integer> matchedIndexes = new ArrayList<Integer>();
        for (Integer index : duelWinners) {
            boolean isIndexMatch = true;
            int textIndex = index;
            for (Character c : pattern) {
                if (c != text[textIndex]) {
                    isIndexMatch = false;
                    break;
                } else {
                    isIndexMatch = true;
                    textIndex++;
                }
            }

            if (isIndexMatch)
                matchedIndexes.add(index);
        }

        return matchedIndexes;
    }

    public ArrayList<Integer> getMatchedPositions(String text) {
        return getMatchedPositions(text.toCharArray());
    }

    private ArrayList<Integer> getDuelWinners(char[] text) {
        ArrayList<Integer> duelWinners = new ArrayList<Integer>();
        for (int i = 0; (i + 1) < text.length - 1; i += 2)
            duelWinners.add(duel(text, i, i + 1));

        return duelWinners;
    }

    private int nextPowerof2(int i) {
        return i == 0 ? 0 : 32 - Integer.numberOfLeadingZeros(i - 1);
    }

    public int treeDuel(String text) {
        return treeDuel(text.toCharArray());
    }

    public int treeDuel(char[] text) {
        return treeDuel(text, 0, text.length);
    }

    public int treeDuel(char[] text, int start, int finish) {
        assert (finish <= text.length);
        assert (finish > start);
        if (finish - start == 1)
            return start;

        // duel is an associative operation, therefore we can use
        // the reduce operator
        OptionalInt result = IntStream.range(start, finish)
                .parallel()
                .reduce((i,j) -> duel(text, i, j));

        return result.getAsInt();
    }

    public static boolean patternMatchesAtPosition(char []txt, char []pat, int startPosition)
    {

        if (pat.length + startPosition > txt.length)
            return false;

        return IntStream.range(0, pat.length)
                .parallel()
                .allMatch(i -> txt[i + startPosition] == pat[i]);
    }

    public static boolean patternMatchesAtPositionParallel(char []txt, char []pat, int startPosition)
    {
        if (pat.length + startPosition > txt.length)
            return false;

        return IntStream.range(0, pat.length)
                        .parallel()
                        .allMatch(i -> txt[i + startPosition] == pat[i]);
    }

    public int[] match(char []text) {
        if (text.length < pattern.length)
            return new int[0];

        int chunkSize = witness.length;

        int[] results = IntStream.range(0, text.length)
                .parallel()
                .filter(i -> i % chunkSize == 0)
                .map(start -> treeDuel(text, start, Math.min(start + chunkSize, text.length)))
                .filter(candidatePosition -> patternMatchesAtPositionParallel(text, pattern, candidatePosition))
                .toArray();

        return results;
    }

    public int[] bruteForceSequentialMatch(char[] text){

        int[] results = IntStream.range(0, text.length - pattern.length)
                .filter(i -> patternMatchesAtPosition(text, pattern, i))
                .toArray();

        return results;
    }

    public int[] match(String text) {
        return match(text.toCharArray());
    }

}