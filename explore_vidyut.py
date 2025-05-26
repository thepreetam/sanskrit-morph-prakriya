import vidyut
print(dir(vidyut))

try:
    from vidyut import prakriya
    print('Successfully imported vidyut.prakriya')
    print(dir(prakriya))
except ImportError:
    print('Could not import vidyut.prakriya')

try:
    from vidyut import args
    print('Successfully imported vidyut.args')
    print(dir(args))
except ImportError:
    print('Could not import vidyut.args')

# Attempt to find Vyakarana and TinantaArgs if they exist at top level
try:
    print(vidyut.Vyakarana)
except AttributeError:
    print('vidyut.Vyakarana not found')

try:
    print(vidyut.TinantaArgs)
except AttributeError:
    print('vidyut.TinantaArgs not found') 