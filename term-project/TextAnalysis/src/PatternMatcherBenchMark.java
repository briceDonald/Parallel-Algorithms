package utexas.parallel;

import com.google.caliper.BeforeExperiment;
import com.google.caliper.Benchmark;
import com.google.caliper.Param;
import com.google.caliper.api.VmOptions;
import com.google.caliper.runner.Running;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;

@VmOptions("-XX:-TieredCompilation")
public class PatternMatcherBenchMark {
    @Param({
            "1000"
    }) int size; // -Dsize=1,2,3

    @Param({
            "1", "2", "3", "4", "5", "6", "7", "8"
    }) int numThreads;
    //String pattern = "PARALLEL";
    String pattern = "Here I present a much longer pattern length to test out in order to see what happens to the execution time for patterns of significanlty longer length.";
    String tString;
    char []tChar;
    PatternMatcher patternMatcher;
    ForkJoinPool customThreadPool;

    @BeforeExperiment
    public void setUp()
    {
        //System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", numThreads);
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < size; i++) {
            builder.append("0123456789");
        }
        builder.append(pattern);
        String tString = builder.toString();
        tChar = tString.toCharArray();
        patternMatcher = new PatternMatcher(pattern);
        customThreadPool = new ForkJoinPool(numThreads);
    }

//    @Benchmark
//    int []bruteForce(int reps) {
//
//        int[] bruteResult = new int[0];
//        for (int i = 0; i < reps; i++) {
//            bruteResult = patternMatcher.bruteForceSequentialMatch(tChar);
//        }
//        return bruteResult;
//    }

    @Benchmark
    int []matcher(int reps) throws ExecutionException, InterruptedException {
        int[] result = new int[0];
        for (int i = 0; i < reps; i++) {
            result = customThreadPool.submit(() -> patternMatcher.match(tChar)).get();
        }
        return result;
    }
}