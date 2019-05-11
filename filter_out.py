import sys

printed_newseg = False
ignore = True

for line in sys.stdin.readlines():
    if line.startswith('---Phase 2---'):
        ignore = False
    if ignore:
        continue
    if line.startswith('pam') or line.startswith('Phase'):
        line = '\n' + line
        sys.stdout.write(line)
    if line.startswith('num'):
        sys.stdout.write(line)
    if line.startswith('STOP'):
        break
    if line.startswith('  learned'):
        print 'lsl was: %s' % line.strip().split()[-1]
