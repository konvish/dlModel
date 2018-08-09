package dl4j;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AnimalModelByHdfsCluster {
    private static final Logger log = LoggerFactory.getLogger(AnimalModelByHdfsCluster.class);
    private static int height = 100; //图片的高度
    private static int width = 100; //图片的宽度
    private static int channels = 3; //channels RGB
    private static long seed = 12345;
    protected static int batchSize = 10;
    protected static int epochs = 50;

    private static String rootPath = System.getProperty("user.dir");
    private static String modelPath = "";
}
