package cn.sibat.gui.java;

import java.awt.*;
import java.awt.event.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;

public class FileDialogDemo implements ActionListener {
    private Frame f;
    private FileDialog fload;
    private FileDialog fsave;
    private TextArea t;
    private String file = "";

    public void init() {
        f = new Frame("My Notepad");
        MenuBar mb = new MenuBar();
        Menu file = new Menu("文件");
        Menu help = new Menu("帮助");
        MenuItem open = new MenuItem("打开");
        MenuItem save = new MenuItem("保存");
        MenuItem saveAs = new MenuItem("另存为");
        file.add(open);
        file.add(save);
        file.add(saveAs);
        mb.add(file);
        mb.add(help);
        f.setMenuBar(mb);
        t = new TextArea();
        f.add(t, "Center");
        open.addActionListener(this);
        save.addActionListener(this);
        saveAs.addActionListener(this);
        f.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
        f.setSize(400, 200);
        f.setLocation(450, 200);
        f.setVisible(true);
        fload = new FileDialog(f, "打开文件", FileDialog.LOAD);
        fsave = new FileDialog(f, "保存文件", FileDialog.SAVE);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        String s = e.getActionCommand();
        switch (s) {
            case "打开":
                fload.setVisible(true);
                String d = fload.getDirectory();
                String f = fload.getFile();
                if (d != null && f != null) {
                    file = d + f;
                    loadFile();
                }
                break;
            case "保存":
                if (file == null || file.equals("")) {
                    this.saveFileAs();
                } else {
                    this.saveFile();
                }
                break;
            case "另存为":
                this.saveFileAs();
                break;
        }
    }

    private void loadFile() {
        t.setText("");
        t.setFont(new Font("隶书", Font.ITALIC, 20));
        t.setForeground(Color.RED);
        f.setTitle("My Notepad-" + file);
        try {
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String s = reader.readLine();
            while (s != null) {
                t.append(s + "\n");
                s = reader.readLine();
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void saveFileAs() {
        fsave.setVisible(true);
        String d = fload.getDirectory();
        String fd = fload.getFile();
        if (d != null && fd != null) {
            file = d + fd;
            this.saveFile();
            f.setTitle("My Notepad-" + file);
        }
    }

    private void saveFile() {
        String text = t.getText();
        try {
            PrintWriter writer = new PrintWriter(new FileWriter(file));
            writer.println(text);
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        new FileDialogDemo().init();
    }
}
