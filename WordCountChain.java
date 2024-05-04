import java.io.IOException;
import java.util.StringTokenizer;
import java.util.regex.Pattern;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.chain.ChainMapper;
import java.util.TreeMap;

public class WordCountChain {

  public static class LowerCaseMapper 
      extends Mapper<Object, Text, IntWritable, Text> {

    private Text lowercased = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        lowercased.set(value.toString().toLowerCase());
        context.write(new IntWritable(1), lowercased);
    }
  }

  public static class RemoveSpecialCharsMapper
       extends Mapper<IntWritable, Text, Text, IntWritable>{

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    private static final Pattern pattern = Pattern.compile("[^a-zA-Z\\s]");

    public void map(IntWritable key, Text value, Context context
                    ) throws IOException, InterruptedException {
      String cleanedLine = pattern.matcher(value.toString()).replaceAll("");
      StringTokenizer itr = new StringTokenizer(cleanedLine);
      while (itr.hasMoreTokens()) {
        word.set(itr.nextToken()); 
        context.write(word, one);
      }
    }
  }

  public static class TokenizerMapper
       extends Mapper<Text, IntWritable, Text, IntWritable>{

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    public void map(Text key, IntWritable value, Context context
                    ) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(key.toString());
      while (itr.hasMoreTokens()) {
        word.set(itr.nextToken()); 
        context.write(word, one);
      }
    }
  }

  public static class IntSumReducer
       extends Reducer<Text,IntWritable,Text,IntWritable> {

    private TreeMap<Text, IntWritable> topN = new TreeMap<Text, IntWritable>();

    public void reduce(Text key, Iterable<IntWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      topN.put(new Text(key), new IntWritable(sum));
      if (topN.size() > 10) {
        topN.remove(topN.firstKey());
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException,
            InterruptedException {
      for (Text i : topN.descendingKeySet()) {
        context.write(i,topN.get(i));
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "word count");
    job.setJarByClass(WordCountChain.class);
    
    Configuration loweCaseMapperConf = new Configuration(false);
    ChainMapper.addMapper(job,
      LowerCaseMapper.class,
      Object.class, Text.class,
      IntWritable.class, Text.class,
      loweCaseMapperConf);

    Configuration removeSpecialCharsMapperConf = new Configuration(false);
    ChainMapper.addMapper(job,
      RemoveSpecialCharsMapper.class,
      IntWritable.class, Text.class,
      Text.class, IntWritable.class,
      removeSpecialCharsMapperConf);

    Configuration tokenizerMapperConf = new Configuration(false);
    ChainMapper.addMapper(job,
      TokenizerMapper.class,
      Text.class, IntWritable.class,
      Text.class, IntWritable.class,
      tokenizerMapperConf);

    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}