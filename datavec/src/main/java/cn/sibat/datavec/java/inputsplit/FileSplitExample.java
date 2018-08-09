package cn.sibat.datavec.java.inputsplit;

import org.datavec.api.split.FileSplit;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.Iterator;
import java.util.Random;

public class FileSplitExample {
    public static void main(String[] args) throws IOException {
        // 加载resources的inputsplit的路径
        ClassPathResource classPathResource1 = new ClassPathResource("inputsplit");
        File directoryToLook = classPathResource1.getFile();

        //=====================================================================
        //                 Example 1: 加载全部文件
        //=====================================================================

        FileSplit fileSplit1 = new FileSplit(directoryToLook);

        System.out.println("--------------- Example 1: Loading every file ---------------");
        URI[] fileSplit1Uris = fileSplit1.locations();
        for (URI uri : fileSplit1Uris) {
            System.out.println(uri);
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //                 Example 2: 加载文件去掉子目录
        //=====================================================================

        FileSplit fileSplit2 = new FileSplit(directoryToLook, null, false);

        System.out.println("--------------- Example 2: Loading non-recursively ---------------");
        URI[] fileSplit2Uris = fileSplit2.locations();
        for (URI uri : fileSplit2Uris) {
            System.out.println(uri);
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //                 Example 3: 加载特定的文件格式
        //=====================================================================

        String[] extensionsToFilter = new String[]{".jpg"};
        FileSplit fileSplit3 = new FileSplit(directoryToLook, extensionsToFilter, false);

        System.out.println("--------------- Example 3: Loading with filters ---------------");
        URI[] fileSplit3Uris = fileSplit3.locations();
        for (URI uri : fileSplit3Uris) {
            System.out.println(uri);
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //                 Example 4: 随机顺序加载文件
        //=====================================================================

        FileSplit fileSplit4 = new FileSplit(directoryToLook, extensionsToFilter, new Random(222));

        System.out.println("--------------- Example 4: Loading with a random seed ---------------");
        Iterator<URI> fileSplit4UrisIterator = fileSplit4.locationsIterator();
        while (fileSplit4UrisIterator.hasNext()) {
            System.out.println(fileSplit4UrisIterator.next());
        }
        System.out.println("------------------------------------------------------------\n\n\n");

        //=====================================================================
        //                 Example 5: 加载单个文件
        //=====================================================================

        FileSplit fileSplit5 = new FileSplit(
                new ClassPathResource("inputsplit/cats/tabby_s_001355.jpg").getFile()
        );

        System.out.println("--------------- Example 5: FileSplit with a single file ---------------");
        Iterator<URI> fileSplit5UrisIterator = fileSplit5.locationsIterator();
        while (fileSplit5UrisIterator.hasNext()) {
            System.out.println(fileSplit5UrisIterator.next());
        }
        System.out.println("------------------------------------------------------------\n\n\n");
    }
}
