// ============================================================================
// bench_progress_dashboard.cpp - the fancy --progress=fancy reporter.
//
// All the TUI machinery (terminal probing, the visible-width-aware Row builder,
// the gradient bar / sparkline primitives) is local to this file; the base only
// provides the case sampling and the plain-line fallback.
// ============================================================================
#include "cli/progress/bench_progress_dashboard.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace rppbench {

namespace {

// --- terminal helpers -------------------------------------------------------

bool stderr_is_tty() {
#if defined(__unix__) || defined(__APPLE__)
    return isatty(fileno(stderr)) != 0;
#else
    return false;
#endif
}

// Current terminal width in columns, re-queried each frame so the dashboard
// reflows on resize. Falls back to 80 when it can't be determined.
int term_cols() {
#if defined(TIOCGWINSZ)
    struct winsize ws{};
    if (ioctl(fileno(stderr), TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return ws.ws_col;
#endif
    return 80;
}

std::string mmss(double seconds) {
    if (!(seconds >= 0.0) || seconds > 359999.0)
        seconds = 0.0;
    int s = static_cast<int>(seconds + 0.5);
    char b[16];
    std::snprintf(b, sizeof(b), "%02d:%02d", s / 60, s % 60);
    return b;
}

// Extract "key:value" (value up to the next '/') from an encoded benchmark name.
std::string name_field(const std::string &name, const char *key) {
    auto p = name.find(key);
    if (p == std::string::npos)
        return {};
    p += std::strlen(key);
    auto e = name.find('/', p);
    return name.substr(p, e == std::string::npos ? std::string::npos : e - p);
}

// --- dashboard rendering primitives -----------------------------------------

// A display line accumulated as raw bytes plus its visible column count, so ANSI
// escapes (zero width) and multibyte glyphs (their own cell count) don't corrupt
// alignment the way byte-based padding would.
struct Row {
    std::string buf;
    int cols = 0;

    void esc(const char *code) { buf += code; } // zero-width control bytes
    void ascii(const std::string &t) {          // one column per byte
        buf += t;
        cols += static_cast<int>(t.size());
    }
    void glyphs(const std::string &g, int cells) { // multibyte, known width
        buf += g;
        cols += cells;
    }
    void pad_to(int target) {
        while (cols < target) {
            buf.push_back(' ');
            ++cols;
        }
    }
};

// 256-color green -> cyan gradient used across the filled portion of the bar.
constexpr std::array<int, 6> kBarPalette{46, 47, 48, 49, 50, 51};

void set_fg(Row &r, int color256) {
    char b[16];
    std::snprintf(b, sizeof(b), "\033[38;5;%dm", color256);
    r.esc(b);
}
void reset(Row &r) {
    r.esc("\033[0m");
}

// A Unicode progress bar `width` cells wide, filled to `frac` with an eighths-
// resolution leading edge and a gradient across the fill.
void render_bar(Row &r, double frac, int width) {
    frac = std::min(std::max(frac, 0.0), 1.0);
    static const char *const kEighths[9] = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
    double exact = frac * width;
    int full = static_cast<int>(exact);
    int rem8 = static_cast<int>((exact - full) * 8.0 + 0.5);
    if (rem8 == 8) {
        ++full;
        rem8 = 0;
    }
    for (int i = 0; i < width; ++i) {
        int gradIdx = full > 1 ? (i * (static_cast<int>(kBarPalette.size()) - 1)) / (full - 1) : 0;
        gradIdx = std::min(gradIdx, static_cast<int>(kBarPalette.size()) - 1);
        if (i < full) {
            set_fg(r, kBarPalette[gradIdx]);
            r.glyphs("█", 1); // full block
            reset(r);
        } else if (i == full && rem8 > 0) {
            set_fg(r, kBarPalette[gradIdx]);
            r.glyphs(kEighths[rem8], 1);
            reset(r);
        } else {
            set_fg(r, 238); // dim track
            r.glyphs("░", 1);
            reset(r);
        }
    }
}

// Sparkline of `vals` scaled between `lo` and `hi`, using block glyphs.
void render_sparkline(Row &r, const std::deque<double> &vals, double lo, double hi) {
    static const char *const kBars[8] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    double span = hi - lo;
    set_fg(r, 45); // cyan-ish
    for (double v : vals) {
        int lvl = span > 1e-12 ? static_cast<int>((v - lo) / span * 7.0 + 0.5) : 0;
        lvl = std::min(std::max(lvl, 0), 7);
        r.glyphs(kBars[lvl], 1);
    }
    reset(r);
}

// Compose one boxed content line: "│ <content padded to CW> │".
std::string boxed(Row &content, int contentWidth) {
    content.pad_to(contentWidth);
    std::string line = "│ ";
    line += content.buf;
    line += " │";
    return line;
}

} // namespace

DashboardProgressReporter::DashboardProgressReporter(int total) : ProgressReporter(total) {
    tty_ = stderr_is_tty();
}

DashboardProgressReporter::~DashboardProgressReporter() {
    if (cursorHidden_) {
        std::fprintf(stderr, "\033[?25h"); // ensure the cursor is restored
        std::fflush(stderr);
    }
}

bool DashboardProgressReporter::ReportContext(const Context &) {
    start_ = Clock::now();
    if (tty_) {
        std::fprintf(stderr, "\033[38;5;51mrpp_bench\033[0m running \033[1m%d\033[0m case(s)\n",
                     total());
        std::fprintf(stderr, "\033[?25l"); // hide cursor while animating
        cursorHidden_ = true;
    } else {
        std::fprintf(stderr, "Running %d benchmark case(s)...\n", total());
    }
    std::fflush(stderr);
    return true;
}

void DashboardProgressReporter::ReportRuns(const std::vector<Run> &runs) {
    Sample s = sample(runs);
    if (s.name.empty())
        return;

    if (s.isNew) {
        // Feed the sparkline and the live tallies once per case.
        recentMs_.push_back(s.ms);
        if (recentMs_.size() == 1)
            minMs_ = maxMs_ = s.ms;
        else {
            minMs_ = std::min(minMs_, s.ms);
            maxMs_ = std::max(maxMs_, s.ms);
        }
        if (const auto b = name_field(s.name, "backend:"); !b.empty())
            ++byBackend_[b];
        if (const auto d = name_field(s.name, "dtype:"); !d.empty())
            ++byDtype_[d];
    }

    if (tty_)
        draw(s.name, s.ms);
    else
        draw_plain_line(done(), total(), s.name, s.val, s.unit);
}

void DashboardProgressReporter::draw(const std::string &name, double lastMs) {
    const int CW = std::min(std::max(term_cols() - 4, 44), 96); // content width
    const double frac = total() > 0 ? static_cast<double>(done()) / total() : 0.0;
    const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
    const double rate = elapsed > 1e-6 ? done() / elapsed : 0.0;
    const double eta = rate > 1e-6 ? (total() - done()) / rate : 0.0;

    std::vector<std::string> lines;

    // Top border with an embedded title.
    {
        std::string top = "╭─ \033[38;5;51mrpp_bench\033[0m ── progress ";
        // Visible cols after '╭': "─ rpp_bench ── progress " -> 1+1+9+1+2+1+8+1.
        // Fill the rest of the CW+2-wide inter-corner span with '─'.
        int used = 1 + 1 + 9 + 1 + 2 + 1 + 8 + 1;
        for (int i = used; i < CW + 2; ++i)
            top += "─";
        top += "╮";
        lines.push_back(std::move(top));
    }

    // Progress bar + percentage + count.
    {
        char rhs[48];
        std::snprintf(rhs, sizeof(rhs), " %5.1f%%  %*d/%d", frac * 100.0,
                      static_cast<int>(std::to_string(total()).size()), done(), total());
        int barW = std::max(CW - static_cast<int>(std::strlen(rhs)) - 1, 10);
        Row r;
        render_bar(r, frac, barW);
        set_fg(r, 250);
        r.ascii(rhs);
        reset(r);
        lines.push_back(boxed(r, CW));
    }

    // Timing: elapsed / ETA / throughput.
    {
        Row r;
        set_fg(r, 244);
        r.ascii("elapsed ");
        reset(r);
        r.ascii(mmss(elapsed));
        set_fg(r, 244);
        r.ascii("   eta ");
        reset(r);
        r.ascii(mmss(eta));
        set_fg(r, 244);
        r.ascii("   rate ");
        reset(r);
        char rt[32];
        std::snprintf(rt, sizeof(rt), "%.2f case/s", rate);
        r.ascii(rt);
        lines.push_back(boxed(r, CW));
    }

    // Current case.
    {
        Row r;
        set_fg(r, 51);
        r.glyphs("▸ ", 2);
        reset(r);
        std::string shown = strip_op_prefix(name);
        int room = CW - r.cols;
        if (room > 1 && static_cast<int>(shown.size()) > room) {
            shown.resize(std::max(room - 1, 0));
            shown += "…"; // ellipsis: 1 display cell
            r.buf += shown;
            r.cols += room; // this content fills the remaining width exactly
        } else {
            r.ascii(shown);
        }
        lines.push_back(boxed(r, CW));
    }

    // Last time + sparkline + spread.
    {
        Row r;
        set_fg(r, 244);
        r.ascii("last ");
        reset(r);
        char last[40]; // normalized to ms so it shares the sparkline/spread scale
        std::snprintf(last, sizeof(last), "%7.3g ms  ", lastMs);
        r.ascii(last);
        int sparkW = std::min(static_cast<int>(recentMs_.size()), std::max(CW - r.cols - 26, 8));
        if (sparkW > 0 && !recentMs_.empty()) {
            std::deque<double> tail(recentMs_.end() - sparkW, recentMs_.end());
            render_sparkline(r, tail, minMs_, maxMs_);
        }
        char spread[40];
        std::snprintf(spread, sizeof(spread), "  min %.3g  max %.3g", minMs_, maxMs_);
        set_fg(r, 244);
        r.ascii(spread);
        reset(r);
        lines.push_back(boxed(r, CW));
    }

    // Backend / dtype tally.
    {
        Row r;
        auto emit = [&r](const char *label, const std::map<std::string, int> &tally) {
            set_fg(r, 244);
            r.ascii(label);
            reset(r);
            for (const auto &[k, v] : tally) {
                char c[48];
                std::snprintf(c, sizeof(c), "%s %d  ", k.c_str(), v);
                r.buf += c;
                r.cols += static_cast<int>(k.size()) + 1 +
                          static_cast<int>(std::to_string(v).size()) + 2; // "%s %d  "
            }
        };
        emit("backend ", byBackend_);
        emit("dtype ", byDtype_);
        lines.push_back(boxed(r, CW));
    }

    // Bottom border.
    {
        std::string bot = "╰";
        for (int i = 0; i < CW + 2; ++i)
            bot += "─";
        bot += "╯";
        lines.push_back(std::move(bot));
    }

    // Redraw in place: move up over the previous frame, rewrite each line.
    if (linesDrawn_ > 0)
        std::fprintf(stderr, "\033[%dA", linesDrawn_);
    for (const auto &l : lines)
        std::fprintf(stderr, "\r%s\033[K\n", l.c_str());
    linesDrawn_ = static_cast<int>(lines.size());
    std::fflush(stderr);
}

void DashboardProgressReporter::Finalize() {
    if (tty_) {
        if (cursorHidden_) {
            std::fprintf(stderr, "\033[?25h");
            cursorHidden_ = false;
        }
        double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
        std::fprintf(stderr, "\033[38;5;46m✓\033[0m done: %d case(s) in %s\n", done(),
                     mmss(elapsed).c_str());
    } else {
        std::fprintf(stderr, "\ndone: %d benchmark case(s).\n", done());
    }
    std::fflush(stderr);
}

} // namespace rppbench
