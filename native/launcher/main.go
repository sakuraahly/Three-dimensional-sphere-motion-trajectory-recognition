// 球形物体三维运动轨迹识别 —— 原生 Windows 启动器（exe）
//
// 为什么需要它（2026-11-05）：`run.bat` 一直是唯一入口，负责挑 Python、
// 装依赖、路由参数；但双击 .bat 会闪一个黑色控制台窗口，也不像"一个程序"。
// 这个 exe 只是薄薄一层壳：
//
//   - 无参数：优先用安装器登记的环境（config/python_env.txt），否则自己找一台
//     **真能 import cv2** 的 Python，用 pythonw.exe 静默启动 GUI；
//   - 一个都没找到：询问是否自动跑 `run.bat install`，装完接着启动 GUI；
//   - 有参数：原样转给 `run.bat`（参数路由的唯一产地仍在 bat，不复制一份）；
//   - 失败：弹系统消息框，而不是窗口一闪而过、看起来像"点了没反应"；
//   - --dry-run：只探测环境、把结果写到 outputs/launcher_probe.txt，不启动任何东西。
//
// 编译（在 native/launcher/ 下双击 build.bat），或手动：
//
//   windres -i icon.rc -o icon.syso -O coff
//   set CGO_ENABLED=0
//   go build -ldflags="-s -w -H windowsgui" -o ..\..\launcher.exe .
//
// 产物是静态原生 PE，不依赖 Go 运行时或额外 DLL。**但它和 .bat 一样需要本机
// 有装好 OpenCV 的 Python** —— 它不是把整个应用打包进去。要"连 Python 都不用装"
// 的独立 exe，得走 PyInstaller 把解释器与全部依赖一起打，那是另一件事（体积数百 MB）。
//
// 注意：Go 会忽略以 `_` 或 `.` 开头的 .go 文件，源码别用下划线开头命名。

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

const appTitle = "球形物体三维运动轨迹识别"

var (
	user32      = syscall.NewLazyDLL("user32.dll")
	messageBoxW = user32.NewProc("MessageBoxW")
)

// alert 用系统消息框报错。-H windowsgui 的程序没有 stdout/stderr，
// 不弹窗的话失败时就是"双击了没反应"（项目里已经踩过这个坑）。
func alert(msg string) {
	title, _ := syscall.UTF16PtrFromString(appTitle)
	text, _ := syscall.UTF16PtrFromString(msg)
	// MB_OK(0) | MB_ICONERROR(0x10) | MB_TOPMOST(0x40000)
	messageBoxW.Call(0, uintptr(unsafe.Pointer(text)),
		uintptr(unsafe.Pointer(title)), 0x40010)
}

// hideWindow 让子进程不弹控制台黑框（CREATE_NO_WINDOW = 0x08000000）。
func hideWindow(c *exec.Cmd) {
	c.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000}
}

// showConsole 给需要用户看见输出的子进程开一个独立控制台
// (CREATE_NEW_CONSOLE = 0x00000010)。本 exe 用 -H windowsgui 编译、自己没有控制台，
// 子进程默认继承不到有效句柄：既看不到 pip 进度，bat 末尾的 pause 也等不到输入。
// 与 hideWindow 对称，只用在「安装 / 带参数命令行」这两类希望用户盯着看的路径。
func showConsole(c *exec.Cmd) {
	c.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x00000010}
}

// fileExists 不是目录且存在。
func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

func rootDir() string {
	exe, err := os.Executable()
	if err != nil {
		wd, _ := os.Getwd()
		return wd
	}
	return filepath.Dir(exe)
}

// pythonCandidates 与 `run.bat` 的 :probe 候选集合一致：
// 用户级 miniforge/anaconda 的 myenv、系统级位置，以及 PATH 里的 python。
// 光看路径不算数 —— msys64 那个 python 在 PATH 里但没装 opencv。
func pythonCandidates() []string {
	var out []string
	add := func(base string, parts ...string) {
		if base == "" {
			return
		}
		out = append(out, filepath.Join(append([]string{base}, parts...)...))
	}
	for _, base := range []string{
		os.Getenv("USERPROFILE"),
		os.Getenv("LOCALAPPDATA"),
		os.Getenv("ProgramData"),
	} {
		for _, distro := range []string{"miniforge3", "anaconda3"} {
			add(base, distro, "envs", "myenv", "python.exe")
		}
	}
	for _, dir := range filepath.SplitList(os.Getenv("PATH")) {
		if dir == "" {
			continue
		}
		out = append(out, filepath.Join(dir, "python.exe"))
		out = append(out, filepath.Join(dir, "python3.exe"))
	}
	return out
}

// probe 真的跑一次 `import cv2, numpy`，通过才算数（与 bat 的 :probe 同口径）。
func probe(py string) bool {
	if py == "" {
		return false
	}
	if st, err := os.Stat(py); err != nil || st.IsDir() {
		return false
	}
	c := exec.Command(py, "-c", "import cv2, numpy")
	hideWindow(c)
	return c.Run() == nil
}

// pointerPython 读安装器登记的环境指针（config/python_env.txt）。
// 纯文本 key=value 而不是 JSON：此刻可能连 Python 都还没配好，不能靠
// Python 来解析；bat 用 for /f、Go 按行切分、安装器用 partition，三方同读一份。
// 项目内的值写的是相对路径，按 root 还原；仓库外的值本身就是绝对路径。
func pointerPython(root string) string {
	b, err := os.ReadFile(filepath.Join(root, "config", "python_env.txt"))
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "python=") {
			continue
		}
		v := strings.TrimSpace(strings.TrimPrefix(line, "python="))
		if v == "" {
			return ""
		}
		if filepath.IsAbs(v) {
			return v
		}
		return filepath.Join(root, filepath.FromSlash(v))
	}
	return ""
}

// ask 弹 Yes/No 框，返回是否点了「是」（IDYES = 6）。
func ask(msg string) bool {
	title, _ := syscall.UTF16PtrFromString(appTitle)
	text, _ := syscall.UTF16PtrFromString(msg)
	// MB_YESNO(4) | MB_ICONQUESTION(0x20) | MB_TOPMOST(0x40000)
	ret, _, _ := messageBoxW.Call(0, uintptr(unsafe.Pointer(text)),
		uintptr(unsafe.Pointer(title)), 0x40024)
	return ret == 6
}

// runInstaller 调 bat 的 install 参数。必须显式开新控制台：pip 下载要几分钟，
// 用户得看得见进度；父进程没有控制台时，bat 末尾的 pause 会立刻返回。
func runInstaller(root string) bool {
	bat := filepath.Join(root, "run.bat")
	if _, err := os.Stat(bat); err != nil {
		return false
	}
	c := exec.Command("cmd", "/c", bat, "install")
	c.Dir = root
	showConsole(c)
	return c.Run() == nil
}

func findPython(root string) (string, int) {
	tried := 0
	if p := pointerPython(root); p != "" {
		tried++
		if probe(p) {
			return p, tried
		}
	}
	for _, py := range pythonCandidates() {
		tried++
		if probe(py) {
			return py, tried
		}
	}
	return "", tried
}

// findGuiScript 优先精确名字，兜底沿用 bat 的做法：
// 扫描根目录 .py，取第一个含 "tkinter" 的（不把中文文件名写死在两处）。
func findGuiScript(root string) string {
	exact := filepath.Join(root, "apps/gui_main.py")
	if st, err := os.Stat(exact); err == nil && !st.IsDir() {
		return exact
	}
	// 兜底扫 apps/ 下含 tkinter 的脚本（正常走不到，留给旧布局）。
	// 不能扫根目录：两个启动脚本搬进 apps/ 后，根目录已经没有 .py 了。
	entries, _ := os.ReadDir(filepath.Join(root, "apps"))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".py") {
			continue
		}
		p := filepath.Join(root, "apps", e.Name())
		b, err := os.ReadFile(p)
		if err == nil && strings.Contains(string(b), "tkinter") {
			return p
		}
	}
	return ""
}

// guiPython 用 pythonw.exe 才不会带控制台；没有就退回 python.exe。
func guiPython(py string) string {
	if py == "" {
		return ""
	}
	w := filepath.Join(filepath.Dir(py), "pythonw.exe")
	if st, err := os.Stat(w); err == nil && !st.IsDir() {
		return w
	}
	return py
}

// batCommand 把参数原样交给 run.bat。bat 自己开控制台并 pause，
// 所以这里统一给子进程开新控制台（test / install / check 等都需要看输出）。
func batCommand(root string, args []string) *exec.Cmd {
	c := exec.Command("cmd", append([]string{"/c", filepath.Join(root, "run.bat")}, args...)...)
	c.Dir = root
	showConsole(c)
	return c
}

func main() {
	root := rootDir()
	args := os.Args[1:]

	// 自检模式：不启动 GUI、不弹窗，把探测结果写文件（也给控制台版打印）。
	if len(args) == 1 && args[0] == "--dry-run" {
		py, tried := findPython(root)
		lines := []string{
			fmt.Sprintf("root       = %s", root),
			fmt.Sprintf("pointer    = %s", pointerPython(root)),
			fmt.Sprintf("python     = %s (tried %d candidates)", py, tried),
			fmt.Sprintf("guiPython  = %s", guiPython(py)),
			fmt.Sprintf("guiScript  = %s", findGuiScript(root)),
			fmt.Sprintf("batCommand = %s", batCommand(root, []string{"test"}).String()),
		}
		text := strings.Join(lines, "\n") + "\n"
		_ = os.MkdirAll(filepath.Join(root, "outputs"), 0755)
		_ = os.WriteFile(filepath.Join(root, "outputs", "launcher_probe.txt"),
			[]byte(text), 0644)
		fmt.Print(text)
		return
	}

	// 带参数：原样交给 bat。bat 会自己开控制台、自己 pause，
	// 命令行工具（test / standards / check ...）需要看到输出。
	if len(args) > 0 {
		bat := filepath.Join(root, "run.bat")
		if _, err := os.Stat(bat); err != nil {
			alert("找不到 run.bat，无法执行参数：" + strings.Join(args, " "))
			return
		}
		// Run() 对子进程非 0 退出会返回 *ExitError，判 err != nil 即可覆盖。
		// 不能再丢弃错误：install 失败时 exe 静默退出，用户会以为"点了没反应"。
		if err := batCommand(root, args).Run(); err != nil {
			alert("命令执行失败：" + strings.Join(args, " ") + "\n\n" +
				"请看日志 logs\\misc\\ 或手动运行 run.bat " + strings.Join(args, " "))
		}
		return
	}

	py, _ := findPython(root)
	if py == "" {
		if !ask("还没有检测到可用的运行环境。\n\n" +
			"要现在自动安装吗？\n" +
			"安装器会复用现成环境，或建一份项目内的隔离 .venv 并装好依赖；\n" +
			"需要联网，可能要几分钟，进度会在控制台窗口里显示。") {
			return
		}
		if !runInstaller(root) {
			alert("安装器没能正常结束。\n\n" +
				"请手动运行：run.bat install\n" +
				"日志目录：logs\\misc\\")
			return
		}
		py, _ = findPython(root)
		if py == "" {
			alert("安装器跑完了，但仍找不到可用环境。\n\n" +
				"请看日志目录：logs\\misc\\")
			return
		}
	}
	// 2026-11-08：**先走 run.bat，只是把它藏起来。**
	//
	// 用户实测（同一台机器、同一份代码、同一个 Python 环境）：
	//   `run.bat`      → 相机预览正常；
	//   `launcher.exe` → 两台相机都「打不开」。
	// 两条路的唯一实质差别在这里：bat 用 `python.exe`（有控制台、走 bat 的环境），
	// 本启动器原来直接起 `pythonw.exe`（无控制台，`sys.stdout is None`）。
	// 既然 bat 是**已验证可用**的那条，启动器就不该再复制一份启动逻辑 ——
	// 它只负责把 bat 的控制台藏起来。没有 run.bat 时才退回直接起 Python。
	if bat := filepath.Join(root, "run.bat"); fileExists(bat) {
		c := exec.Command("cmd", "/c", bat, "gui")
		c.Dir = root
		hideWindow(c)
		if err := c.Start(); err == nil {
			return
		}
	}

	script := findGuiScript(root)
	if script == "" {
		alert("在 " + root + " 里找不到图形界面脚本（apps/gui_main.py）。")
		return
	}

	c := exec.Command(guiPython(py), script)
	c.Dir = root
	hideWindow(c)
	if err := c.Start(); err != nil {
		alert("启动图形界面失败：\n" + err.Error())
		return
	}
	// 故意不 Wait：exe 立刻退出，GUI 作为独立进程继续运行。
}
