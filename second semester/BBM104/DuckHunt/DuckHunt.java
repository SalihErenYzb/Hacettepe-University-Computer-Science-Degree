import javafx.application.Application;
import javafx.scene.image.*;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.*;
import javafx.stage.Stage;
/**

The DuckHunt class is the main class that launches the game.
It extends the Application class from JavaFX library and overrides its start method.
It also contains static final variables for scaling the game's graphics, setting the volume, and defining the images used in the game.
It creates the crosshair cursor used in the game and updates it with the updateCursor method.
It also creates a MediaPlayer object and plays the title music on a loop.
The class has a GameStateManager object that is used to manage the state of the game.
The main method launches the application.
The challenging part of the code is managing the state of the game and changing between the MenuScene and GameScene when necessary.
This is achieved using a listener that observes the current state of the GameStateManager object and changes the scene accordingly.
The updateCursor method also presents a challenge in scaling the crosshair cursor images properly.
**/
public class DuckHunt extends Application {
	public static final double SCALE =3;
	public static final double VOLUME =0.025;
	public static Image[] Backgrounds = {new Image("assets/background/1.png"),new Image("assets/background/2.png"),
			new Image("assets/background/3.png"),new Image("assets/background/4.png"),
			new Image("assets/background/5.png"),new Image("assets/background/6.png")};
	public static Image[] Foregrounds ={new Image("assets/foreground/1.png"),new Image("assets/foreground/2.png"),
			new Image("assets/foreground/3.png"),new Image("assets/foreground/4.png"),
			new Image("assets/foreground/5.png"),new Image("assets/foreground/6.png")};
	public static String[] CrosshairsStr ={"assets/crosshair/1.png","assets/crosshair/2.png",
			"assets/crosshair/3.png","assets/crosshair/4.png",
			"assets/crosshair/5.png","assets/crosshair/6.png",
			"assets/crosshair/7.png"};
	public static Image[] Crosshairs =new Image[CrosshairsStr.length];
	private static int i = 0;
	private static int j = 0;
	private static ImageCursor crosshairCursor;
	public static void updateCursor() {
		for (int a = 0; a < CrosshairsStr.length; a++) {
			Image originalImage = new Image(CrosshairsStr[a]);
			double scaledWidth = originalImage.getWidth() * SCALE;
			double scaledHeight = originalImage.getHeight() * SCALE;
			Image scaledImage = new Image(CrosshairsStr[a], scaledWidth, scaledHeight, true, false);
			Crosshairs[a] = scaledImage;
		}
		DuckHunt.crosshairCursor = new ImageCursor(Crosshairs[j], Crosshairs[j].getWidth() / 2,
				Crosshairs[j].getHeight() / 2);
	}
	
	
	public static MediaPlayer mediaPlayer ;

	private static GameStateManager gameStateManager = new GameStateManager();
	static Stage primaryStage;
    @Override
    public void start(Stage primaryStage) {
    	updateCursor();
    	DuckHunt.primaryStage = primaryStage;
        // Load the favicon image
        Image favicon = new Image("assets/favicon/1.png");
        
        
        //Start with menu
        primaryStage.setScene(new MenuScene()); 
        
        Media media = new Media(getClass().getResource("assets/effects/Title.mp3").toExternalForm());
        
        // create a MediaPlayer object with the Media object for Title and BackGround 
        mediaPlayer = new MediaPlayer(media);
        mediaPlayer.setVolume(DuckHunt.VOLUME);

        mediaPlayer.setVolume(DuckHunt.VOLUME);
        mediaPlayer.setCycleCount(MediaPlayer.INDEFINITE);

        mediaPlayer.play();
        
        // Change scenes accordingly beetween gameScene and MenuScene
	    DuckHunt.getGameStateManager().currentStateProperty()
	    .addListener((observable, oldState, newState) -> {
	    	if (newState == GameStateManager.GameState.LEVEL1) {
	    		mediaPlayer.stop();
	    		primaryStage.setScene(new GameScene());
	    	}else if (newState == GameStateManager.GameState.TITLE) {
	    		primaryStage.setScene(new MenuScene());
	    		mediaPlayer.play();
	    	}
	    });
        
	    
	    
	    
	    
        primaryStage.setResizable(false); // Set resizable property to false

        primaryStage.getIcons().add(favicon);
        primaryStage.setTitle("HUBBM Duck Hunt");
        primaryStage.show();
    }
    
    public static void main(String[] args) {
    	
        launch(args);
    }

	public static int getI() {
		return i;
	}

	public static void setI(int i) {
		DuckHunt.i = i;
		
	}

	
	public static int getJ() {
		return j;
	}

	public static void setJ(int j) {
		DuckHunt.j = j;
	}

	public static GameStateManager getGameStateManager() {
		return gameStateManager;
	}

	public static ImageCursor getCrosshairCursor() {
		return crosshairCursor;
	}

	public static void setCrosshairCursor(ImageCursor crosshairCursor) {
		DuckHunt.crosshairCursor = crosshairCursor;
	}
	public static Stage Itself() {
		return primaryStage;
	}

}