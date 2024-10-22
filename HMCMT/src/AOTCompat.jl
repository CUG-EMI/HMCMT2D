const _is_win = Sys.iswindows()

function unix_readline(f, windows_line=_is_win)::Union{String, Nothing}
    line = Char[]
    try
        while true
            c = read(f, Char)
            if windows_line && c == '\r'
                c = read(f, Char)
                @assert c == '\n' "read $c after newline"
                break
            end
            c == '\n' && break
            push!(line, c)
        end
        return String(line)
    catch e
        e isa EOFError && begin
            return String(line)
        end
        throw(e)
    end
end


function _time_seconds()
    ccall(:time, UInt, (Ptr{Cvoid},), C_NULL)
end

macro compat_elapsed(ex)
    quote
        local t0 = _time_seconds()
        local val = $(esc(ex))
        local t1 = _time_seconds()
        (float(t1 - t0), val)
    end
end