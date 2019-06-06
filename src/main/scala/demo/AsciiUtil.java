package demo;

/**
 * AsciiUtil
 * @author liuji
 *
 */
public class AsciiUtil {

    private static final char SBC_SPACE = 12288;
    private static final char DBC_SPACE = 32;
    private static final char ASCII_START = 33;
    private static final char ASCII_END = 126;
    private static final char UNICODE_START = 65281;
    private static final char UNICODE_END = 65374;
    private static final char DBC_SBC_STEP = 65248;
    //全角转半角
    public static String sbc2dbcCase(String src) {
        if (src == null) {
            return null;
        }
        char[] c = src.toCharArray();
        for (int i = 0; i < c.length; i++) {
            c[i] = sbc2dbc(c[i]);
        }
        return new String(c);
    }
    //半角转全角
    public static String dbc2sbcCase(String src) {
        if (src == null) {
            return null;
        }

        char[] c = src.toCharArray();
        for (int i = 0; i < c.length; i++) {
            c[i] = dbc2sbc(c[i]);
        }
        return new String(c);
    }
    private static char sbc2dbc(char src){
        if (src == SBC_SPACE) {
            return DBC_SPACE;
        }
        if (src >= UNICODE_START && src <= UNICODE_END) {
            return (char) (src - DBC_SBC_STEP);
        }
        return src;
    }
    private static char dbc2sbc(char src){
        if (src == DBC_SPACE) {
            return SBC_SPACE;
        }
        if ((src <= ASCII_END) && (src >= ASCII_START)) {
            return (char) (src + DBC_SBC_STEP);
        }
        return src;
    }
}