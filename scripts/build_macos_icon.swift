import AppKit

let args = CommandLine.arguments
guard args.count == 2 else {
    fputs("Usage: build_macos_icon.swift <output-png-path>\n", stderr)
    exit(1)
}

let outputURL = URL(fileURLWithPath: args[1])
let size = CGSize(width: 1024, height: 1024)
let rect = CGRect(origin: .zero, size: size)

guard let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil,
    pixelsWide: Int(size.width),
    pixelsHigh: Int(size.height),
    bitsPerSample: 8,
    samplesPerPixel: 4,
    hasAlpha: true,
    isPlanar: false,
    colorSpaceName: .deviceRGB,
    bytesPerRow: 0,
    bitsPerPixel: 0
) else {
    fputs("Could not create bitmap image rep\n", stderr)
    exit(1)
}

NSGraphicsContext.saveGraphicsState()
guard let context = NSGraphicsContext(bitmapImageRep: rep) else {
    fputs("Could not create graphics context\n", stderr)
    exit(1)
}
NSGraphicsContext.current = context

let roundedRect = NSBezierPath(roundedRect: rect.insetBy(dx: 48, dy: 48), xRadius: 220, yRadius: 220)
roundedRect.addClip()

let bgGradient = NSGradient(colors: [
    NSColor(calibratedRed: 0.21, green: 0.11, blue: 0.10, alpha: 1.0),
    NSColor(calibratedRed: 0.42, green: 0.19, blue: 0.15, alpha: 1.0),
    NSColor(calibratedRed: 0.58, green: 0.29, blue: 0.21, alpha: 1.0)
])
bgGradient?.draw(in: roundedRect, angle: -40)

let glowRect = CGRect(x: 120, y: 500, width: 780, height: 420)
let glow = NSGradient(starting: NSColor(calibratedRed: 0.93, green: 0.78, blue: 0.64, alpha: 0.28), ending: NSColor.clear)
glow?.draw(in: NSBezierPath(ovalIn: glowRect), relativeCenterPosition: .zero)

let gridPath = NSBezierPath()
gridPath.lineWidth = 3
NSColor(calibratedWhite: 1.0, alpha: 0.10).setStroke()
for index in 0...6 {
    let x = 140 + CGFloat(index) * 120
    gridPath.move(to: CGPoint(x: x, y: 180))
    gridPath.line(to: CGPoint(x: x, y: 844))
}
for index in 0...6 {
    let y = 180 + CGFloat(index) * 110
    gridPath.move(to: CGPoint(x: 140, y: y))
    gridPath.line(to: CGPoint(x: 884, y: y))
}
gridPath.stroke()

struct Cell {
    let center: CGPoint
    let radius: CGFloat
    let fill: NSColor
    let stroke: NSColor
    let nucleus: NSColor
}

let cells: [Cell] = [
    Cell(center: CGPoint(x: 310, y: 645), radius: 110, fill: NSColor(calibratedRed: 0.98, green: 0.64, blue: 0.44, alpha: 0.88), stroke: NSColor(calibratedRed: 1.0, green: 0.92, blue: 0.85, alpha: 0.75), nucleus: NSColor(calibratedRed: 0.43, green: 0.11, blue: 0.14, alpha: 0.9)),
    Cell(center: CGPoint(x: 500, y: 700), radius: 130, fill: NSColor(calibratedRed: 0.41, green: 0.83, blue: 0.73, alpha: 0.86), stroke: NSColor(calibratedRed: 0.90, green: 1.0, blue: 0.96, alpha: 0.80), nucleus: NSColor(calibratedRed: 0.07, green: 0.23, blue: 0.28, alpha: 0.92)),
    Cell(center: CGPoint(x: 690, y: 610), radius: 118, fill: NSColor(calibratedRed: 0.95, green: 0.79, blue: 0.35, alpha: 0.88), stroke: NSColor(calibratedRed: 1.0, green: 0.97, blue: 0.88, alpha: 0.78), nucleus: NSColor(calibratedRed: 0.34, green: 0.20, blue: 0.07, alpha: 0.92)),
    Cell(center: CGPoint(x: 390, y: 420), radius: 104, fill: NSColor(calibratedRed: 0.42, green: 0.63, blue: 0.96, alpha: 0.82), stroke: NSColor(calibratedRed: 0.90, green: 0.95, blue: 1.0, alpha: 0.78), nucleus: NSColor(calibratedRed: 0.06, green: 0.17, blue: 0.44, alpha: 0.92)),
    Cell(center: CGPoint(x: 620, y: 390), radius: 112, fill: NSColor(calibratedRed: 0.88, green: 0.50, blue: 0.72, alpha: 0.80), stroke: NSColor(calibratedRed: 1.0, green: 0.91, blue: 0.98, alpha: 0.78), nucleus: NSColor(calibratedRed: 0.32, green: 0.08, blue: 0.22, alpha: 0.92))
]

for cell in cells {
    let cellPath = NSBezierPath(ovalIn: CGRect(x: cell.center.x - cell.radius, y: cell.center.y - cell.radius, width: cell.radius * 2, height: cell.radius * 2))
    cell.fill.setFill()
    cellPath.fill()

    cellPath.lineWidth = 12
    cell.stroke.setStroke()
    cellPath.stroke()

    let nucleusRadius = cell.radius * 0.24
    let nucleusOffset = CGPoint(x: cell.radius * 0.16, y: -cell.radius * 0.12)
    let nucleusPath = NSBezierPath(ovalIn: CGRect(
        x: cell.center.x + nucleusOffset.x - nucleusRadius,
        y: cell.center.y + nucleusOffset.y - nucleusRadius,
        width: nucleusRadius * 2,
        height: nucleusRadius * 2
    ))
    cell.nucleus.setFill()
    nucleusPath.fill()
}

let boundaryPath = NSBezierPath()
boundaryPath.lineWidth = 16
let dashPattern: [CGFloat] = [28, 18]
boundaryPath.setLineDash(dashPattern, count: dashPattern.count, phase: 0)
boundaryPath.move(to: CGPoint(x: 215, y: 250))
boundaryPath.curve(to: CGPoint(x: 835, y: 720), controlPoint1: CGPoint(x: 340, y: 520), controlPoint2: CGPoint(x: 600, y: 215))
NSColor(calibratedRed: 0.98, green: 0.99, blue: 1.0, alpha: 0.88).setStroke()
boundaryPath.stroke()

let scanRing = NSBezierPath(ovalIn: CGRect(x: 700, y: 180, width: 160, height: 160))
scanRing.lineWidth = 14
NSColor(calibratedRed: 0.95, green: 0.99, blue: 0.96, alpha: 0.92).setStroke()
scanRing.stroke()

let scanDot = NSBezierPath(ovalIn: CGRect(x: 752, y: 232, width: 56, height: 56))
NSColor(calibratedRed: 0.97, green: 0.72, blue: 0.34, alpha: 0.96).setFill()
scanDot.fill()

let outerBorder = NSBezierPath(roundedRect: rect.insetBy(dx: 48, dy: 48), xRadius: 220, yRadius: 220)
outerBorder.lineWidth = 10
NSColor(calibratedWhite: 1.0, alpha: 0.18).setStroke()
outerBorder.stroke()

NSGraphicsContext.restoreGraphicsState()

guard let pngData = rep.representation(using: .png, properties: [:]) else {
    fputs("Could not encode PNG\n", stderr)
    exit(1)
}

do {
    try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try pngData.write(to: outputURL)
} catch {
    fputs("Failed to write icon PNG: \(error)\n", stderr)
    exit(1)
}

print("Wrote \(outputURL.path)")
