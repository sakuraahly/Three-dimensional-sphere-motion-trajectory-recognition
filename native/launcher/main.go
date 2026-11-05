// 球形物体三维运动轨迹识别 —— 原生 Windows 启动器（exe）
//
// 为什么需要它（2026-11-05）：`启动程序.bat` 一直是唯一入口，负责挑 Python、
// 装依赖、路由参数；但双击 .bat 会闪一个黑色控制台窗口，也不像"一个程序"。
// 这个 exe 只是薄薄一层壳：
//
//   - 无参数：自己找一台**真能 import cv2** 的 Python，用 pythonw.exe 静默启动 GUI；
//   - 有参数：原样转给 `启动程序.bat`（参数路由的唯一产地仍在 bat，不复制一份）；
//   - 失败：弹系统消息框，而不是窗口一闪而过、看起来像"点了没反应"；
//   - --dry-run：只探测环境、把结果写到 outputs/launcher_probe.txt，不启动任何东西。
//
// 编译（在 native/launcher/ 下双击 build.bat），或手动：
//
//   windres -i icon.rc -o icon.syso -O coff
//   set CGO_ENABLED=0
//   go build -ldflags="-s -w -H windowsgui" -o ..\..\启动器.exe .
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

func rootDir() string {
	exe, err := os.Executable()
	if err != nil {
		wd, _ := os.Getwd()
		return wd
	}
	return filepath.Dir(exe)
}

// pythonCandidates 与 `启动程序.bat` 的 :probe 候选集合一致：
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

func findPython() (string, int) {
	tried := 0
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
	exact := filepath.Join(root, "启动程序.py")
	if st, err := os.Stat(exact); err == nil && !st.IsDir() {
		return exact
	}
	entries, _ := os.ReadDir(root)
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".py") {
			continue
		}
		p := filepath.Join(root, e.Name())
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

func batCommand(root string, args []string) *exec.Cmd {
	c := exec.Command("cmd", append([]string{"/c", filepath.Join(root, "启动程序.bat")}, args...)...)
	c.Dir = root
	return c
}

func main() {
	root := rootDir()
	args := os.Args[1:]

	// 自检模式：不启动 GUI、不弹窗，把探测结果写文件（也给控制台版打印）。
	if len(args) == 1 && args[0] == "--dry-run" {
		py, tried := findPython()
		lines := []string{
			fmt.Sprintf("root       = %s", root),
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
		bat := filepath.Join(root, "启动程序.bat")
		if _, err := os.Stat(bat); err != nil {
			alert("找不到 启动程序.bat，无法执行参数：" + strings.Join(args, " "))
			return
		}
		_ = batCommand(root, args).Run()
		return
	}

	py, _ := findPython()
	if py == "" {
		alert("没有找到装好 OpenCV / NumPy 的 Python。\n\n" +
			"请先双击 启动程序.bat，它会询问是否自动安装依赖；\n" +
			"或手动执行：pip install -r requirements.txt")
		return
	}
	script := findGuiScript(root)
	if script == "" {
		alert("在 " + root + " 里找不到图形界面脚本（启动程序.py）。")
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
