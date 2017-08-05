package utexas.parallel;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;

public class PatternMatcherTest {

    @Test
    public void WitnessIsCorrectLengthForNonPeriodicPattern() {
        PatternMatcher matcher = new PatternMatcher("abcde");
        Assert.assertEquals(3, matcher.witness.length);

        matcher = new PatternMatcher("abcdef");
        Assert.assertEquals(3, matcher.witness.length);

        matcher = new PatternMatcher("abcdefg");
        Assert.assertEquals(4, matcher.witness.length);

        matcher = new PatternMatcher("aaabbcddeeedf");
        Assert.assertEquals(7, matcher.witness.length);

        matcher = new PatternMatcher("abcaabcab");
        Assert.assertEquals(5, matcher.witness.length);

        matcher = new PatternMatcher("aaabaaa");
        Assert.assertEquals(4, matcher.witness.length);

        // Special case: We expect at least 2 elements, since we're doing 0 based arrays.
        matcher = new PatternMatcher("ab");
        Assert.assertEquals(2, matcher.witness.length);
    }

    @Test
    public void WitnessIsCorrectLengthForPeriodicPattern() {
        PatternMatcher matcher;

        matcher = new PatternMatcher("aaabaaabaaaba");
        Assert.assertEquals(4, matcher.witness.length);

        matcher = new PatternMatcher("xyzxyzxyz");
        Assert.assertEquals(3, matcher.witness.length);

        matcher = new PatternMatcher("11111221111122");
        Assert.assertEquals(7, matcher.witness.length);

        matcher = new PatternMatcher("111222111222111");
        Assert.assertEquals(6, matcher.witness.length);

    }

    @Test
    public void WitnessHasCorrectValues() {
        String text = "";
        PatternMatcher matcher = new PatternMatcher("aaabbcddeeedf");
        ValidateWitness(matcher.pattern, matcher.witness);
    }

    private void ValidateWitness(char[] pattern, int[] witness) {
        for (int offset = 0; offset < witness.length; offset++) {
            int differingPatternIndex = witness[offset];

            if (differingPatternIndex == -1) {
                for (int i = 0; i < witness.length; i++) {
                    Assert.assertEquals(pattern[i], pattern[i - offset]);
                }
            } else {
                Assert.assertTrue(pattern[differingPatternIndex] != pattern[differingPatternIndex + offset]);
            }
        }
    }

    @Test
    public void DuelReturnsCorrectWinner() {
        // Example from 7.5 from Jaja
        String text = "abcaabcabaa";
        String pattern = "abcabcab";
        // Note that Jaja's original example (witness={0,3,2}, i=5, j=7)
        // uses 1-based indexing and thus will not work with our algorithm unless we convert to
        // 0 based indexing
        int [] witness = {0, 2, 1};
        int i = 4;
        int j = 6;
        PatternMatcher matcher = new PatternMatcher(pattern, witness);

        Assert.assertEquals(i, matcher.duel(text, i, j));
    }

    @Test
    public void GetMatchedPositions_NoMatchesFound()
    {
        String text = "abjasodfa";
        String pattern = "abed";

        PatternMatcher matcher = new PatternMatcher(pattern);
        Assert.assertTrue(matcher.getMatchedPositions(text).isEmpty());
    }

    @Test
    public void GetMatchedPositions_SingleMatchFound()
    {
        String text = "eeeabedkkkk";
        String pattern = "abed";

        PatternMatcher matcher = new PatternMatcher(pattern);
        ArrayList<Integer> matches =  matcher.getMatchedPositions(text);
        Assert.assertEquals(1, matches.size());
        Assert.assertEquals(3, (int)matches.get(0));

        text = "eeeeeeeeeeeeeeeabedkkkk";
        pattern = "abed";
        matcher = new PatternMatcher(pattern);
        matches =  matcher.getMatchedPositions(text);
        Assert.assertEquals(1, matches.size());
        Assert.assertEquals(15, (int)matches.get(0));
    }

    @Test
    public void GetMatchedPositions_MultipleMatchesFound()
    {
        String text = "eeeabedkkkkabedeeeeabedkkkkk";
        String pattern = "abed";

        PatternMatcher matcher = new PatternMatcher(pattern);
        ArrayList<Integer> matches =  matcher.getMatchedPositions(text);
        Assert.assertEquals(3, matches.size());
        Assert.assertEquals(3, (int)matches.get(0));
        Assert.assertEquals(11, (int)matches.get(1));
        Assert.assertEquals(19, (int)matches.get(2));
    }

    @Test
    public void testPeriodicity()
    {
        Assert.assertTrue(PatternMatcher.getPeriod("aaabaaab").equals("aaab"));
        Assert.assertTrue(PatternMatcher.getPeriod("ababab").equals("ab"));
        Assert.assertTrue(PatternMatcher.getPeriod("abcabc").equals("abc"));
        Assert.assertTrue(PatternMatcher.getPeriod("aaa").equals("a"));
        Assert.assertTrue(PatternMatcher.getPeriod("ababababa").equals("ab"));
        Assert.assertTrue(PatternMatcher.getPeriod("abcabca").equals("abc"));
        Assert.assertTrue(PatternMatcher.getPeriod("xxxyyz").equals("xxxyyz"));
    }

    @Test
    public void testTreeDuelEx77()
    {
        /* Test's Jaja's example 7.7 */
        String T = "babaababaaba";
        String P = "abaab";
        int[] witness = {-1, 0, 1};

        PatternMatcher patternMatcher = new PatternMatcher(P, witness);
        Assert.assertEquals(1, patternMatcher.treeDuel(T.toCharArray(), 0, 3));
        Assert.assertEquals(4, patternMatcher.treeDuel(T.toCharArray(), 3, 6));
        Assert.assertEquals(6, patternMatcher.treeDuel(T.toCharArray(), 6, 9));
        Assert.assertEquals(9, patternMatcher.treeDuel(T.toCharArray(), 9, 12));
    }

    @Test
    public void testTreeDuel()
    {
        String P = "123";

        PatternMatcher patternMatcher = new PatternMatcher(P);
        //Assert.assertEquals(0, patternMatcher.treeDuel("12".toCharArray(), 0, 2));

        // chunks are of size ceil(m/2)=2, so we must treeDuel on the two chunks
        // and not the entire array
        Assert.assertEquals(1, patternMatcher.treeDuel("x123".toCharArray(), 0, 3));
        Assert.assertEquals(2, patternMatcher.treeDuel("x123".toCharArray(), 2, 4));
    }


    @Test
    public void testTreeDuelEx77NoWitness()
    {
        String T = "babaababaaba";
        String P = "abaab";

        PatternMatcher patternMatcher = new PatternMatcher(P);
        Assert.assertEquals(1, patternMatcher.treeDuel(T.toCharArray(), 0, 3));
        Assert.assertEquals(4, patternMatcher.treeDuel(T.toCharArray(), 3, 6));
        Assert.assertEquals(6, patternMatcher.treeDuel(T.toCharArray(), 6, 9));
        Assert.assertEquals(9, patternMatcher.treeDuel(T.toCharArray(), 9, 12));

    }

    @Test
    public void testNonPeriodicMatch_noMatchesFound()
    {
        String T = "babaababaaba";
        String P = "cde";
        PatternMatcher patternMatcher = new PatternMatcher(P);
        Assert.assertTrue(patternMatcher.match(T.toCharArray()).length == 0);
    }

    @Test
    public void testNonPeriodicMatch()
    {
        // Jaja Example 7.7
        String T = "babaababaaba";
        String P = "abaab";
        int[] witness = {-1, 0, 1};
        PatternMatcher patternMatcher = new PatternMatcher(P, witness);
        Assert.assertArrayEquals(new int[]{1, 6}, patternMatcher.match(T.toCharArray()));

        P = "PARALLEL";
        patternMatcher = new PatternMatcher(P);

        Assert.assertArrayEquals(new int[]{0, 8}, patternMatcher.match("PARALLELPARALLEL"));
        Assert.assertArrayEquals(new int[]{3, 11}, patternMatcher.match("xxxPARALLELPARALLEL"));
        Assert.assertArrayEquals(new int[]{6}, patternMatcher.match("012345PARALLEL012345"));
        Assert.assertArrayEquals(new int[]{2}, patternMatcher.match("xxPARALLELyy"));
        Assert.assertArrayEquals(new int[]{2}, patternMatcher.match("xxPARALLEL"));
        Assert.assertArrayEquals(new int[]{1}, patternMatcher.match("xPARALLEL"));
        Assert.assertArrayEquals(new int[]{2}, patternMatcher.match("  PARALLEL "));
        Assert.assertArrayEquals(new int[]{21}, patternMatcher.match("0123456789x123456789xPARALLEL"));
        Assert.assertArrayEquals(new int[]{37}, patternMatcher.match("0123456789x123456789x123456789x123456PARALLEL"));
        Assert.assertArrayEquals(new int[]{38}, patternMatcher.match("0123456789x123456789x123456789x1234567PARALLEL"));
        Assert.assertArrayEquals(new int[]{47}, patternMatcher.match("This is a relatively long string with the word PARALLEL in it"));
        Assert.assertArrayEquals(new int[]{}, patternMatcher.match("This is a relatively long string with the word a typo in the word PARALLLEL in it"));

        patternMatcher = new PatternMatcher("AB");
        Assert.assertArrayEquals(new int[]{1}, patternMatcher.match("xAB"));
        Assert.assertArrayEquals(new int[]{0}, patternMatcher.match("AB"));
    }


    @Test
    public void testBenchFind() {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            builder.append("0123456789");
            builder.append("PARALLEL");
        }

        String t = builder.toString();

        PatternMatcher pm = new PatternMatcher("PARALLEL");
        pm.bruteForceSequentialMatch(t.toCharArray());
    }

    @Test
    public void testBenchFindParallel() {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            builder.append("0123456789");
            builder.append("PARALLEL");
        }
        String t = builder.toString();
        PatternMatcher pm = new PatternMatcher("PARALLEL");
        pm.match(t);
    }

    @Test
    public void testFailure()
    {
        String P = "AB";
        PatternMatcher patternMatcher = new PatternMatcher(P);

    }

    @Test
    public void testMemCmpOffset()
    {
        Assert.assertEquals(true, PatternMatcher.patternMatchesAtPosition("abchectorabd".toCharArray(), "hector".toCharArray(), 3));
        Assert.assertEquals(false, PatternMatcher.patternMatchesAtPosition("abchectorabd".toCharArray(), "hector".toCharArray(), 4));
        Assert.assertEquals(true, PatternMatcher.patternMatchesAtPosition("foobar".toCharArray(), "foo".toCharArray(), 0));
    }

    @Test
    public void testBruteForceSequentialMatch()
    {
        String T = "babaababaaba";
        String P = "abaab";
        int[] witness = {-1, 0, 1};
        PatternMatcher patternMatcher = new PatternMatcher(P, witness);
        Assert.assertArrayEquals(new int[]{1, 6}, patternMatcher.bruteForceSequentialMatch(T.toCharArray()));
    }
}