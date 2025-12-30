# synthesize_all.tcl - Production-ready timing extraction (v4.0 FINAL)

set output_file "timing_results.csv"
set fp [open $output_file "w"]
puts $fp "design_name,gate_count,net_count,logic_depth,fanout_max,fanout_avg,critical_path_delay_ns,slack_ns,clock_period_ns"

set designs {
    adder_4bit_ripple
    adder_8bit_ripple
    adder_4bit_cla
    multiplier_4x4
    mux_4to1
    decoder_3to8
    encoder_8to3
    comparator_8bit
    alu_4bit
    parity_generator_8bit
    counter_4bit_up
    counter_8bit_updown
    shift_register_8bit
    fsm_sequence_detector
    traffic_light_controller
    fifo_8x4
    lfsr_8bit
    register_file_4x8
    pwm_generator
    gray_counter_4bit
}

foreach design $designs {
    puts "\n=========================================="
    puts "Processing: $design"
    puts "=========================================="
    
    # Initialize all variables with default values
    set gate_count 0
    set net_count 0
    set logic_depth 0
    set fanout_max 0
    set fanout_avg 0.0
    set delay 0.0
    set slack 0.0
    set clock_period 10.0
    
    # Create project
    catch {file delete -force ./temp_proj}
    create_project -force temp_proj ./temp_proj -part xc7a35tcpg236-1
    
    # Add source file
    catch {add_files "./${design}.v"}
    catch {set_property top $design [current_fileset]}
    
    # Synthesize
    catch {synth_design -top $design -part xc7a35tcpg236-1}
    
    # ===== FEATURE EXTRACTION =====
    
    # 1. Gate Count
    catch {
        set gate_count [llength [get_cells -hierarchical -filter {REF_NAME != VCC && REF_NAME != GND}]]
    }
    
    # 2. Net Count
    catch {
        set net_count [llength [get_nets -hierarchical]]
    }
    
    # 3. Logic Depth
    catch {
        set timing_paths [get_timing_paths -max_paths 1]
        if {[llength $timing_paths] > 0} {
            set logic_depth [get_property LOGIC_LEVELS [lindex $timing_paths 0]]
        }
    }
    
    # 4. Fanout Statistics (SAFE)
    set fanout_list [list]
    catch {
        set net_list [get_nets -hierarchical]
        foreach net $net_list {
            catch {
                set pins [get_pins -of_objects $net -filter {DIRECTION == IN}]
                set fanout [llength $pins]
                if {$fanout > 0} {
                    lappend fanout_list $fanout
                }
            }
        }
    }
    
    # Calculate fanout stats
    if {[llength $fanout_list] > 0} {
        set fanout_max [lindex [lsort -integer -decreasing $fanout_list] 0]
        set fanout_sum 0
        foreach fo $fanout_list {
            set fanout_sum [expr {$fanout_sum + $fo}]
        }
        set fanout_avg [expr {double($fanout_sum) / double([llength $fanout_list])}]
    } else {
        set fanout_max 0
        set fanout_avg 0.0
    }
    
    # 5. Timing Path Delay and Slack (SAFE)
    catch {
        set timing_paths [get_timing_paths -max_paths 1]
        if {[llength $timing_paths] > 0} {
            set path [lindex $timing_paths 0]
            set delay_raw [get_property DATAPATH_DELAY $path]
            set slack_raw [get_property SLACK $path]
            
            if {$delay_raw ne ""} {
                set delay [expr {double($delay_raw)}]
            }
            if {$slack_raw ne ""} {
                set slack [expr {double($slack_raw)}]
            }
        }
    }
    
    # ===== WRITE TO CSV =====
    # Build CSV line with safe format
    set csv_line [format "%s,%d,%d,%d,%d,%.2f,%.4f,%.4f,%.2f" \
        $design \
        $gate_count \
        $net_count \
        $logic_depth \
        $fanout_max \
        $fanout_avg \
        $delay \
        $slack \
        $clock_period]
    
    puts $fp $csv_line
    
    # Print to console
    puts "  Gate Count:        $gate_count"
    puts "  Net Count:         $net_count"
    puts "  Logic Depth:       $logic_depth"
    puts "  Fanout Max:        $fanout_max"
    puts "  Fanout Avg:        $fanout_avg"
    puts "  Delay (ns):        $delay"
    puts "  Slack (ns):        $slack"
    puts "  Clock Period (ns): $clock_period"
    puts "  ✓ CSV entry written"
    
    # Close project
    catch {close_project}
    catch {file delete -force ./temp_proj}
}

close $fp

puts "\n=========================================="
puts "✓ SYNTHESIS COMPLETE"
puts "=========================================="
puts "Output file: timing_results.csv"
puts "Location: [pwd]/timing_results.csv"
puts "=========================================="
