// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

// Window dimensions — must match tauri.conf.json width/height.
// We use hardcoded values because outer_size() returns (0,0) at setup time.
const WIN_W: f64 = 380.0;
const WIN_H: f64 = 720.0; // Reduced from 760 to fit smaller Linux screens better

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            use tauri::Manager;
            if let Some(window) = app.get_webview_window("main") {
                if let Ok(Some(monitor)) = window.primary_monitor() {
                    let work_area = monitor.work_area();
                    let scale_factor = monitor.scale_factor();

                    // Convert work area to logical pixels.
                    let wa_x = work_area.position.x as f64 / scale_factor;
                    let wa_y = work_area.position.y as f64 / scale_factor;
                    let wa_w = work_area.size.width as f64 / scale_factor;
                    let wa_h = work_area.size.height as f64 / scale_factor;

                    // Calculate safe height (don't exceed work area height minus padding)
                    let safe_padding_bottom = 80.0_f64; // Increased to clear larger docks/taskbars
                    let safe_padding_right = 24.0_f64;
                    
                    let actual_h = WIN_H.min(wa_h - safe_padding_bottom - 20.0);
                    
                    // Set size explicitly to ensure it matches our calculation
                    window.set_size(tauri::LogicalSize::new(WIN_W, actual_h)).unwrap();

                    let x = (wa_x + wa_w - WIN_W - safe_padding_right).max(wa_x);
                    let y = (wa_y + wa_h - actual_h - safe_padding_bottom).max(wa_y);

                    window.set_position(tauri::LogicalPosition::new(x, y)).unwrap();
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
