
import re
import sys

fishh = open('fish.h', 'r')
prototype = re.compile(r"\b([a-z\s]+)\s+(fish_\w+)(.*)")
enumdef = re.compile(r"\s*FISH_(\w+).*")

func_proto = """
static int _%(funcname)s(lua_State *L)
{
  %(getargs)s
  %(call)s
  %(push)s
}"""

fbodies = [ ]
wrapped = [ ]
enums = [ ]

for line in fishh:
    if line.startswith('typedef') or line.startswith('struct'): continue
    if line.startswith('char *'):
        line = line.replace('char *', 'charstar ')

    if '//' in line:
        line = line[:line.index('//')-1]

    m = prototype.match(line[:-2])
    if not m:
        n = enumdef.match(line)
        if n:
            enums.append(n.groups())
        continue
    retval, funcname, argstr = m.groups()

    args = argstr[1:-1].split(',')
    if not args[0].startswith('fish'):
        continue
    getargs = [ ]
    argnames = [ ]
    narg = 1
    for arg in args:
        argtype, argname = arg.split()[:-1], arg.split()[-1]
        argtype = ''.join(argtype)
        if argname.startswith('**'):
            argname = argname[2:]
            argtype += ' **'
        elif argname.startswith('*'):
            argname = argname[1:]
            argtype += ' *'

        ga = None

        if argtype == "fish_descr *":
            ga = """fish_descr *%s = *((fish_descr**) """\
                """luaL_checkudata(L, %d, "fish::descr"));""" % (argname, narg)
        elif argtype == "fish_state *":
            ga = """fish_state *%s = *((fish_state**) """\
                """luaL_checkudata(L, %d, "fish::state"));""" % (argname, narg)
        elif argtype == "fish_riemn *":
            ga = """fish_riemn *%s = *((fish_riemn**) """\
                """luaL_checkudata(L, %d, "fish::riemn"));""" % (argname, narg)
        elif argtype == "fluids_state **":
            print argname
            ga = """fluids_state **%s = (fluids_state**) lua_touserdata(L, %d);"""\
                % (argname, narg)
        elif argtype == "long":
            ga = """long %s = luaL_checklong(L, %d);""" % (argname, narg)
        elif argtype == "int":
            ga = """int %s = luaL_checkinteger(L, %d);""" % (argname, narg)
        elif argtype == "double":
            ga = """int %s = luaL_checknumber(L, %d);""" % (argname, narg)
        elif argtype == "char *":
            ga = """char *%s = (char*)luaL_checkstring(L, %d);""" % (argname, narg)
        elif argtype == "int *":
            ga = """int *%s = (int*) lua_touserdata(L, %d);""" % (argname, narg)
        elif argtype == "double *":
            ga = """double *%s = (double*) lua_touserdata(L, %d);""" % (argname, narg)
        elif argtype == "double **":
            ga = """double **%s = (double**) lua_touserdata(L, %d);""" % (argname, narg)
        elif argtype == "void *":
            ga = """void *%s = lua_touserdata(L, %d);""" % (argname, narg)
        else:
            ga = """void *%s = lua_touserdata(L, %d);""" % (argname, narg)

        if ga == None: ga = "unknown " + argtype
        getargs.append(ga)
        argnames.append(argname)
        narg += 1

    call = funcname + '(' + ', '.join(argnames) + ');'

    if retval == 'void':
        push = "return 0;"
    elif retval == 'charstar':
        call = "char *ret = " + call
        push = "lua_pushstring(L, ret);\n  return 1;"
    else:
        call = ("%s ret = " % retval) + call
        push = "lua_pushnumber(L, ret);\n  return 1;"
        
    fbodies.append(func_proto % {'funcname': funcname,
                                 'getargs': '\n  '.join(getargs),
                                 'call': call,
                                 'push': push})
    wrapped.append(funcname)


wrap = open('fishfuncs.c', 'w')

for fbody in fbodies:
    wrap.write(fbody)

wrap.write("\nstatic luaL_Reg fish_module_funcs[] = {\n")
for f in wrapped:
    wrap.write("""    {"%s", _%s},\n""" % (f[7:], f))
wrap.write("    {NULL, NULL}};\n")

wrap.write("static void register_constants(lua_State *L)\n{\n")
for m in enums:
    wrap.write("""  lua_pushnumber(L, FISH_%s); lua_setfield(L, -2, "%s");\n""" % (
            m[0], m[0]))
wrap.write("}\n")