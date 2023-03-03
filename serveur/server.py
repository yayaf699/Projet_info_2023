import cherrypy 

class MonSiteWeb(object):                   # Classe maitresse de l'App 
    def index(self):                        # Méthode invoquée comme URL racine (/)
        return """
            <form action="salutation" method="GET">
            Bonjour. Quel est votre nom ?
            <input type="text" name="nom" />
            <input type="submit" value="OK" />
            </form>
            """
    index.exposed = True                    # la méthode doit être 'publiée'
    def salutation(self, nom = None):
        if nom:
            return "Bonjour {0}, comment allez-vous?".format(nom)
        else:
            return 'Entrez votre nom <a href="/">ici</a>'
    salutation.exposed = True

###### Programme principal : ########
cherrypy.quickstart(MonSiteWeb(), config="server.conf")
