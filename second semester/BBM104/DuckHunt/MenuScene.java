import javafx.scene.input.KeyCode;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.util.Duration;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Scene;
/**

This class represents the menu scene of the game.
It extends the Scene class and provides functionality
for controlling the title and background.
*/
public class MenuScene extends Scene {
	private Media media2 = new Media(getClass().
			getResource("assets/effects/Intro.mp3").toExternalForm());
    private MediaPlayer mediaPlayer2 = new MediaPlayer(media2);
    
    
	
    public MenuScene() {
    	super(new TitleGroup() ,DuckHunt.SCALE*256,DuckHunt.SCALE*240);
        mediaPlayer2.setVolume(DuckHunt.VOLUME);

        
        //Control beetween title and background
	    DuckHunt.getGameStateManager().currentStateProperty()
	    .addListener((observable, oldState, newState) -> {
	    	if (newState == GameStateManager.GameState.TITLE) {

	    		this.setRoot(new TitleGroup());
	    	}else if (newState == GameStateManager.GameState.BACKGROUND) {
	    		
	    		this.setRoot( new BackGroup());
	    		
	    	}
	    });
	    
	    
	    EventHandler<ActionEvent> eventHandler2 = e -> {
	    	DuckHunt.mediaPlayer.stop();
	        mediaPlayer2.play();
	    	};
	    EventHandler<ActionEvent> eventHandler3 = e -> {
	    		DuckHunt.getGameStateManager()
	    		.setCurrentState(GameStateManager.GameState.LEVEL1);
		   };
		   
	    	// control crosshair and background change also going back to title are leaving platform

	    this.setOnKeyPressed(e ->{

	    	if (e.getCode() == KeyCode.ENTER && 
	    			DuckHunt.getGameStateManager().getCurrentState()
	    			== GameStateManager.GameState.TITLE) {
	    		DuckHunt.setI(0);
	    		DuckHunt.setJ(0);
	    		DuckHunt.getGameStateManager()
	    		.setCurrentState(GameStateManager.GameState.BACKGROUND);

	    	}else if (e.getCode() == KeyCode.ENTER && 
	    			DuckHunt.getGameStateManager().getCurrentState()
	    			== GameStateManager.GameState.BACKGROUND) {
	    		Timeline musicAnimation = new Timeline(new KeyFrame(
	    				Duration.millis(mediaPlayer2.getMedia()
	    						.getDuration().toMillis()-600),eventHandler3));
	    		eventHandler2.handle(null);
	    		musicAnimation.play();
	    		

	    	}else if (e.getCode() == KeyCode.ESCAPE && 
	    			DuckHunt.getGameStateManager().getCurrentState()
	    			== GameStateManager.GameState.TITLE) {
                Platform.exit();

	    	}else if (e.getCode() == KeyCode.ESCAPE && 
	    			DuckHunt.getGameStateManager().getCurrentState()
	    			== GameStateManager.GameState.BACKGROUND) {
	    		DuckHunt.getGameStateManager()
	    		.setCurrentState(GameStateManager.GameState.TITLE);

	    	}else if (e.getCode() == KeyCode.UP && 
	    			DuckHunt.getGameStateManager().getCurrentState()
	    			== GameStateManager.GameState.BACKGROUND) {
	    		if (DuckHunt.getJ() < 6) {DuckHunt.setJ(DuckHunt.getJ()+1);}
	    		else {DuckHunt.setJ(0);}	    		
	    		this.setRoot(new BackGroup());

	    	}else if (e.getCode() == KeyCode.DOWN && 
	    			DuckHunt.getGameStateManager().getCurrentState()
	    			== GameStateManager.GameState.BACKGROUND) {
	    		if (DuckHunt.getJ() > 0) {DuckHunt.setJ(DuckHunt.getJ()-1);}
	    		else {DuckHunt.setJ(6);}
	    		this.setRoot(new BackGroup());
	    	}else if (e.getCode() == KeyCode.RIGHT && 
	    			DuckHunt.getGameStateManager().getCurrentState()
	    			== GameStateManager.GameState.BACKGROUND) {
	    		if (DuckHunt.getI() < 5) {DuckHunt.setI(DuckHunt.getI()+1);}
	    		else {DuckHunt.setI(0);}	    		
	    		this.setRoot(new BackGroup());

	    	}else if (e.getCode() == KeyCode.LEFT && 
	    			DuckHunt.getGameStateManager().getCurrentState()
	    			== GameStateManager.GameState.BACKGROUND) {
	    		if (DuckHunt.getI() > 0) {DuckHunt.setI(DuckHunt.getI()-1);}
	    		else {DuckHunt.setI(5);}
	    		this.setRoot(new BackGroup());
	    	}
	    	
	    });
        

    }


}